import os
from pickletools import string1
import shutil
import numpy as np
import cv2
import math
# import pytesseract
# from pytesseract import Output
from scipy import ndimage
from genericpath import isdir
from scipy import misc
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import time
import string
import keras_ocr
reg = keras_ocr.detection.Detector()
image_folder = "./enhanced_images/inverse_law/test_dataset.mp4"
files = os.listdir(image_folder)
print(files)
path = files[0]
p = image_folder + "/" + path
print(p)
image = keras_ocr.tools.read(p)
img_det = reg.detect([image])
# print(img_det[0])
img_tup = []
for i in range(len(img_det[0])):
    st = 'roi'+str(i+1)
    img_tup.append((st, img_det[0][i]))
# print(img_tup)
img = keras_ocr.tools.drawBoxes(image, img_det[0], color=(
    36, 255, 12), thickness=2, boxes_format='boxes')
# print(img)
# img = keras_ocr.tools.drawAnnotations(image=image, predictions=img_tup,ax=None)
# print(img)
# fig, axs = plt.subplots(nrows=1, figsize=(20, 20))
# keras_ocr.tools.drawAnnotations(
#         image=image, predictions=img_tup, ax=axs)
# plt.show()
# print(img_tup[0][1])
# print(img_tup[0][1][0])
# print(img_tup[0][1][0][0])
for i in img_tup:
    x=int(i[1][0][0])
    y = int(i[1][0][1])
    # print(f"{x}   -  {y}")
    cv2.putText(img, i[0], (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 49, 49), 2)
cv2.imshow('window_name', img)
r=input("Please select the ROIs on which you want OCR to be done by entering them with comma as a separator:")
rl=r.split(",")
rm=list(map(lambda x:"roi"+x,rl))
rf=tuple(filter(lambda x:x[0] in rm,img_tup))
print(rf)
ROIs = cv2.selectROIs(
    "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img)
# print rectangle points of selected roi
# ROIs.append(img_det)
print(ROIs)
# names = []
ROI_tup=[]
for i in ROIs:
    name = input(
        'Please enter the attribute name for the selected ROIS:')
    # names.append(name)
    ROI_tup.append((name,i))
ROI_tup.extend(rf)
# print(ROI_tup)
for name,coords in ROI_tup:
    print(f'{name} --- {coords}  {type(coords)} {coords.shape}')
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# # closing all open windows
cv2.destroyAllWindows()


