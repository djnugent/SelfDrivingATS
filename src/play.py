from capnctrl import cap, ctrl
import time
import winsound
import math
import cv2
import numpy as np
import argparse
import datetime
import os
import imageio
import threading
import requests
import serial
import struct
from keras.models import load_model
from model import mean_precision_error, extract_camera, extract_minimap


#*********************************************************************
#-------------------------------- Params -----------------------------
#*********************************************************************
# Mouse bounds
bounds = [600,1000]
# Keyboard steering
step = 0.015
decay = 0.01
# Wheel steering
steering_expo = 3
# Main loop max frequency
update_rate = 30
# Cruise control pid
kP = 0.1
kI = 0.01
kFF = 0.0089
max_int = 0.15
# How fast to transition headlight patterns
flash_rate = 1 #hz
# Recording
record_rate = 4 # hz
reaction_time = 0.315 # seconds
record_delay = 2 # seconds
# ATS telemetry server
API_URL = "http://127.0.0.1:25555/api/ets2/telemetry"

#*********************************************************************
#---------------------------- State Variables ------------------------
#*********************************************************************
steering = 0
throttle = 0
i_term = 0
cruise_speed = 0
cruise_control = False
autopilot = False
recording = False
last_update = 0
last_record = 0
last_flash = 0
light_state = 0
fps = 20


#*********************************************************************
#-------------------------------- Functions --------------------------
#*********************************************************************

def record_entry(_timestamp,_screen,_steering,_throttle,_speed,_speed_limit,_fps,_cruise_control):
    global recording
    global steering
    # delay to grab mouse to account for reaction time
    time.sleep(reaction_time)
    _steering = steering
    #mouse = cap.mouse()
    #_steering = max(-1,min((mouse[0] - 800)/200.0,1))
    # Delay saving entry in order to shave off the last # seconds on recording
    # This is here incase we crash and dont want the last few seconds
    time.sleep(record_delay)
    if recording:
        utc_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S.%f")
        filename = "ats-" + utc_datetime + ".png"
        # Write image
        imageio.imwrite(args.dir + "/" + filename,_screen,compression=1)
        # Add entry to text file
        txt = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(_timestamp,filename,_steering,_throttle,_speed,_speed_limit,_fps,_cruise_control)
        labels.write(txt)
        labels.flush()

#*********************************************************************
#-------------------------------- Start up ---------------------------
#*********************************************************************
parser = argparse.ArgumentParser(description='Run ATS interface')
parser.add_argument('dir', help='Dataset record directory')
args = parser.parse_args()

# Create virtual joystick
ctrl.create_joystick()

# Connect to steering wheel
wheel = serial.Serial('COM5',9600)


# initialize dataset log
if not os.path.exists(args.dir):
    os.mkdir(args.dir)
if not os.path.exists(args.dir+"/labels.csv"):
    labels = open(args.dir + "/labels.csv",'a')
    labels.write("timestamp,img_file,steering,throttle,speed,speed_limit,fps,cruise_control\n")
else:
    labels = open(args.dir + "/labels.csv",'a')


# model
print("loading model...")
model = load_model("../pretrained_models/model.h5",custom_objects={'mean_precision_error': mean_precision_error})
print("done")

# Wait for ATS telemetry server
print("Waiting for ATS telemetry server...")
while True:
    try:
        requests.get(API_URL)
        break
    except requests.exceptions.ConnectionError:
        time.sleep(0.4)

# Wait for game to open
if requests.get(API_URL).json()["game"]["connected"] == False:
    print("Please start ATS...")
    while requests.get(API_URL).json()["game"]["connected"] == False:
        time.sleep(0.4)

# wait to enter the game
print("Press G to start")
while not 'g' in cap.keyboard():
    time.sleep(0.01)
# Center mouse upon starting
ctrl.mouse((800,450))

#*********************************************************************
#-------------------------------- Main Loop --------------------------
#*********************************************************************
while True:
    if time.time()-last_update > 1.0/update_rate:
        dt = time.time()-last_update
        fps = fps * 0.9 + (1.0/dt * 0.1) #EMA
        last_update = time.time()

        # Capture Data
        keys = cap.keyboard()
        mouse = cap.mouse()
        timestamp = time.time()
        try:
            screen = cap.screen(window="American Truck Simulator", padding =[6,6,28,6])
        except ValueError:
            continue

        # Get speed/speed_limit from ATS telemetry API
        _data = requests.get(API_URL).json()
        speed = _data["truck"]["speed"] / 1.609344 #kph -> mph
        speed_limit = round(_data["navigation"]["speedLimit"]  / 1.609344) #kph -> mph

        # Check for change in control(autopilot vs manual)
        if "home" in keys:
            if autopilot:
                autopilot = False
                time.sleep(0.2)
                cap.keyboard()
            else:
                autopilot = True
                recording = False
                time.sleep(0.2)
                cap.keyboard()

        # autopilot mode
        if autopilot:
            camera,roi = extract_camera(screen)
            minimap,roi = extract_minimap(screen)
            steering = float(model.predict([camera[None,:,:,:],minimap[None,:,:,:]], batch_size=1))
            throttle = 0.5
        # manual mode
        else:
            ## Control steering with keyboard
            '''
            # Steer Left
            if 'a' in keys:
                steering = max(steering - step,-1)
            # Steer Right
            elif 'd' in keys:
                steering = min(steering + step,1)
            # Recenter
            else:
                if steering > 0:
                    steering = max(0,steering-decay)
                elif steering < 0:
                    steering = min(0,steering+decay)
            '''
            ## Control steering with mouse
            '''
            if mouse[0] < bounds[0]:
                ctrl.mouse((bounds[0],mouse[1]))
            if mouse[0] > bounds[1]:
                ctrl.mouse((bounds[1],mouse[1]))

            steering = max(-1,min((mouse[0] - 800)/200.0,1))
            '''
            ## Control steering with wheel
            # Request steering angle
            wheel.write(struct.pack('b',0x45))
            wheel.flush()
            # Read angle
            steering = struct.unpack('<H',wheel.read(2))[0] / 511.5 - 1
            # Apply expo
            steering = abs(pow(steering,steering_expo))/steering



            ## Cruise Control
            # Check for enable
            if 'c' in keys and not cruise_control:
                cruise_control = True
                # set speed
                if speed_limit != 0:
                    cruise_speed = speed_limit
                else:
                    cruise_speed = speed
                i_error = 0.5
            # Run cruise controller
            if cruise_control:
                ## Check user Input
                # Increase speed
                if "w" in keys:
                    cruise_speed = min(cruise_speed + 5, 100)
                # Decrease speed
                if "s" in keys:
                    cruise_speed = max(0,cruise_speed - 5)
                # Set to speed limit
                if 'c' in keys and speed_limit != 0:
                    cruise_speed = speed_limit
                # Disable cruise control
                if "x" in keys:
                    cruise_control = False

                # PI Controller
                spd_error = cruise_speed - speed
                i_error = max(-max_int,min(i_error + spd_error * dt,max_int))
                p_term = max(-0.35,kP * spd_error)
                i_term = kI * i_error
                ff_term = kFF * cruise_speed
                throttle = max(-1,min(p_term + i_term + ff_term,1))


            ## Manual throttle control
            else:
                # Forward
                if "w" in keys:
                    throttle = 1
                # Reverse/Brake
                elif "s" in keys:
                    throttle = -1
                # Coast
                else:
                    throttle = 0


            # Change record state
            if 'r' in keys:
                # Stop recording
                if recording:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                    recording = False
                    time.sleep(0.2)
                    cap.keyboard()
                # Start recording
                else:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    recording = True
                    time.sleep(0.2)
                    cap.keyboard()

            if recording:
                # Record Entry in another thread in order to capture user reaction time and drop the last few frames
                if time.time() - last_record > 1.0/record_rate:
                    last_record = time.time()
                    threading.Thread(target=record_entry, args = (timestamp,screen,steering,throttle,speed,speed_limit,fps,cruise_control)).start()

                # Flash lights for various lighting patterns
                if time.time() - last_flash > 1.0/flash_rate:
                    last_flash = time.time()
                    transition_keys = ['ll','k','l']
                    transition = transition_keys[light_state%3]
                    light_state += 1
                    for k in transition:
                        ctrl.key_press(k)
                        time.sleep(0.03)

        # Command joystick
        ctrl.joystick_sticks(left_stick_x=steering,left_stick_y=throttle)

        ## Debug
        if autopilot:
            mode = "AUTOPILOT"
        elif recording:
            mode = "RECORDING"
        else:
            mode = "MANUAL"
        if cruise_control:
            c_speed = cruise_speed
        else:
            c_speed = "DISABLED"
        fmt = "[ {} ] fps: {}, steering: {}, throttle: {}, speed: {}, speed limit: {}, cruise control: {}"
        print(fmt.format(mode,round(fps),round(steering,3),round(throttle,2),round(speed,1),speed_limit,c_speed))
