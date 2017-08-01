import imageio
import glob

files = glob.glob("E:/dataset/*.png")
print(len(files),"Files found")
i = 0
for f in files:
    print(i,f)
    img = imageio.imread(f)
    imageio.imwrite(f,img,compression=9)
    i+=1
