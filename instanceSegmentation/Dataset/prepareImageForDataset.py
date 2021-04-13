# this file change each pixel above the threshold

import cv2
import os
import numpy as np
DATASET_FOLDER= "dataset/val/"
DATASET_OUTPUT= "dataset/val/"

list_images= os.listdir(DATASET_FOLDER)

for images in list_images:
    # This is used to fill pixel inside mask 
    # dilation and erode step
    print(images)
    img = cv2.imread(DATASET_FOLDER + images,0)
    kernel = np.ones((5,5), np.uint8)
    # this remove near pixel not inside the mask
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    ret,thresh1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)

    cv2.imwrite(DATASET_FOLDER+images,thresh1)

    # img = cv2.imread(DATASET_FOLDER + images)
    # image_name= os.path.splitext(images)[0]
    # img = cv2.resize(img,(550,550))
    # cv2.imwrite(DATASET_FOLDER+image_name+".png",img)


