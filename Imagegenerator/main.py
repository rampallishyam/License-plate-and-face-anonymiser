"""
This script takes video in mp4 format as the input and 
generates frames

"""

import cv2
import os

cam = cv2.VideoCapture('video/input_video1.mp4')

try:
    if not os.path.exists('frames'):
        os.mkdir('frames')

except:
    print("Error in creating the directory")

currentframe = 0
filename=1

while(True):

    ret,frame = cam.read()

    if ret:

        name = './frames/' + str(filename) + '.jpg'

        # print ('Creating....'+ name)

        if currentframe%345 ==0:
            print ('Creating....'+ name)
            cv2.imwrite(name,frame)
            filename+=1
        else:
            pass

        currentframe+=1
    
    else:
        break

cam.release()
cv2.destroyAllWindows()
