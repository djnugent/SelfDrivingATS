from capnctrl import cap, ctrl
import time
import winsound
import math
import cv2
from speed import speed
import numpy as np
import argparse
import datetime
import os
import imageio
import threading
from keras.models import load_model
from train import preprocess
from model import mean_precision_error

parser = argparse.ArgumentParser(description='Run ATS interface')
parser.add_argument('dir', help='Dataset record directory')
args = parser.parse_args()

# Speed parser
spd_ocr = speed()
spd_ocr.load()
# Create virtual joystick
ctrl.create_joystick()

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
model = load_model("model.h5",custom_objects={'mean_precision_error': mean_precision_error})
print("done")


# Params
# Mouse bounds
bounds = [600,1000]
# Keyboard steering
step = 0.015
decay = 0.01
# Main loop max frequency
update_rate = 30
# Cruise control pid
kP = 0.1
kI = 0.01
kFF = 0.0089
max_int = 0.15
# How fast to transition headlight patterns
flash_rate = 0.4
# Recording buffer size
buffer_size = 40
# Record rate
record_rate = 3

# State variable
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
buff = []


def hud_speed(img):
    #img = img[428:435,775:787]
    img = img[492:499,779:791]
    spd= spd_ocr.predict(img)
    return spd

def hud_speed_limit(img):
    #img = img[529:536,773:785]
    img = img[593:600,777:789]
    # Check to see that white speed limit icon is present
    if np.mean(img) > 130:
        img = (225 - img) #invert image
        spd_limit = spd_ocr.predict(img)
        return spd_limit
    return None


# wait to enter the game
print("Press G to start")
while not 'g' in cap.keyboard():
    time.sleep(0.01)
# Center mouse upon starting
ctrl.mouse((800,450))

## Main loop
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
            #cv2.imshow("img",screen)
            #cv2.waitKey(0)
        except ValueError:
            continue
        speed = hud_speed(screen)
        speed_limit = hud_speed_limit(screen)

        # Check for change in control(autopilot vs manual)
        if "home" in keys:
            if autopilot:
                autopilot = False
                time.sleep(0.2)
                cap.keyboard()
            else:
                autopilot = True
                time.sleep(0.2)
                cap.keyboard()

        # autopilot mode
        if autopilot:
            img = preprocess(screen)
            steering = float(model.predict(img[None, :, :, :], batch_size=1))
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
            if mouse[0] < bounds[0]:
                ctrl.mouse((bounds[0],mouse[1]))
            if mouse[0] > bounds[1]:
                ctrl.mouse((bounds[1],mouse[1]))
            steering = max(-1,min((mouse[0] - 800)/200.0,1))


            ## Cruise Control
            # Check for enable
            if 'c' in keys and not cruise_control:
                cruise_control = True
                # set speed
                if speed_limit is not None:
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
                if 'c' in keys and speed_limit is not None:
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
                    buff = []
                    time.sleep(0.2)
                    cap.keyboard()

            if recording:
                if time.time() - last_record > 1.0/record_rate:
                    last_record = time.time()
                    # Create a new entry in buffer
                    entry = [timestamp,screen,steering,throttle,speed,speed_limit,fps,cruise_control]
                    buff = [entry] + buff[:buffer_size-1]
                    # Save oldest entry in buffer to account for delay of when I crash to when I stop recording
                    try:
                        _timestamp,_screen,_steering,_throttle,_speed,_speed_limit,_fps,_cruise_control = buff[buffer_size-1]
                        utc_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S.%f")
                        filename = "ats-" + utc_datetime + ".png"
                        # Write image
                        threading.Thread(target=imageio.imwrite, args = (args.dir + "/" + filename,_screen), kwargs={'compression':0}).start()
                        # Add entry to text file
                        txt = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(_timestamp,filename,_steering,_throttle,_speed,_speed_limit,_fps,_cruise_control)
                        labels.write(txt)
                        labels.flush()
                    except IndexError:
                        pass

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
        print(fmt.format(mode,round(fps),round(steering,2),round(throttle,2),speed,speed_limit,c_speed))
