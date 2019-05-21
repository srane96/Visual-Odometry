# =============================================================================================================================================================================================================================== #
#-------------> Project 05 <---------------#
# =============================================================================================================================================================================================================================== #
# Course    :-> ENPM673 - Perception for Autonomous Robots
# Date      :-> 03 May 2019
# Authors   :-> Niket Shah(UID: 116345156), Siddhesh(UID: 116147286), Sudharsan(UID: 116298636)
# =============================================================================================================================================================================================================================== #

# =============================================================================================================================================================================================================================== #
# Import Section for Importing library
# =============================================================================================================================================================================================================================== #
import time, sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import math, glob
from scipy import stats
from ReadCameraModel import *
from UndistortImage import *

# ======================================================================================================================================================================= #
# Function for Importing Images from coressponding folders
# ======================================================================================================================================================================= #
def import_images(foldername: str)-> dict:
    images = np.array([cv.imread(img,0) for img in glob.glob(foldername+'/*.png')])
    return images

# ======================================================================================================================================================================= #
# Function to write video to the folders
# ======================================================================================================================================================================= #
def video_writer(data: list, name: str, frames_per_sec: int)-> None:
    width = data[0].shape[1]
    height = data[0].shape[0]

    print('width:',width,'height:',height)
    print(data[0].shape)

    video = cv.VideoWriter('Data/'+name+'.avi', cv.VideoWriter_fourcc(*'XVID'), frames_per_sec, (width, height))

    for key_frame in data:
        video.write(key_frame)
        
    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    t1 = time.time()
    images = import_images('Data/stereo/centre')
    color = []
    undistorted_image = []
    undistorted_image_gray = []
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('D:/Sudharsan/Academics/Semester 02/ENPM673 - Perception/Projects/Project 05/Data/model')
    for i, img in enumerate(images):
        color_img = cv.cvtColor(img, cv.COLOR_BAYER_GR2BGR)
        color.append(color_img)
        undistorted_image.append(UndistortImage(color_img,LUT))
        undistorted_image_gray.append(cv.cvtColor(UndistortImage(color_img,LUT), cv.COLOR_BGR2GRAY))


    video_writer(undistorted_image[30:], 'undistorted_data', 23)
    video_writer(undistorted_image_gray[30:], 'undistorted_data_gray', 23)
    t2 = time.time() 
    print(len(color))
    cv.imshow("bayer",images[20])
    cv.imshow("Color",color[20])
    cv.imshow("Undist",undistorted_image[20])
    cv.imshow("Undist",undistorted_image_gray[20])
    cv.waitKey(0)


    print("TIme Elapsed: ", t2 -t1)
