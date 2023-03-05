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
# from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt
import time
import string
import easyocr
# import keras_ocr
# reg = keras_ocr.detection.Detector()
image_folder = "./enhanced_images/inverse_law/Stable_Video_1.mp4"
files = os.listdir(image_folder)
print(files)
path = files[0]
p = image_folder + "/" + path
print(p)
img=cv2.imread(p)
reader = easyocr.Reader(['en'])
# image = keras_ocr.tools.read(p)
img_det = reader.detect(img)
print(img_det)
print(img_det[0])
print(img_det[0][0])
print(img_det[0][0][0])
ctr=0
img_tup = []
for i in img_det:
    for j in i[0]:
        # print(type(j[0]))
        if(isinstance( j[0], np.integer)):
            print(j)
            ctr+=1
            st = 'roi'+str(ctr)
            img_tup.append((st, j))
        if (isinstance(j[0], list)):
            print(j)
            ctr += 1
            st = 'roi'+str(ctr)
            img_tup.append((st, j))
print(img_tup)
# for i in img_det:
#     print(len(i))
#     for j in i:
#         print(len(j))
#         for k in j:
#             print(f'{k}  {type(k[0])}')
# for i in range(len(img_det))
# img_tup = []
# for i in range(len(img_det[0])):
#     st = 'roi'+str(i+1)
#     img_tup.append((st, img_det[0][i]))
# # print(img_tup)
# img = keras_ocr.tools.drawBoxes(image, img_det[0], color=(
#     36, 255, 12), thickness=2, boxes_format='boxes')
# print(img)
# img = keras_ocr.tools.drawAnnotations(image=image, predictions=img_tup,ax=None)
# print(img)
# fig, axs = plt.subplots(nrows=1, figsize=(20, 20))
# keras_ocr.tools.drawAnnotations(
#         image=image, predictions=img_tup, ax=axs)
# plt.show()
# txt=reader.readtext(p)
# for i in txt:
#     print(i)
print(img_tup[0][1])
print(img_tup[0][1][0])
for i in img_tup:
    if (isinstance(i[1][0], np.integer)):
        x1=int(i[1][0])
        y1 = int(i[1][3])
        x2 = int(i[1][1])
        y2 = int(i[1][2])
        # print(f"{x}   -  {y}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, i[0], (x1, y1+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 49, 49), 2)
    else:
        
        pts=i[1]
        pts = np.array(pts)
        x=int(pts[0][0])
        y=int(pts[0][1])
        print(f"{x}   -  {y}")
        pts = pts.reshape((-1, 1, 2))
        image = cv2.polylines(img, np.int32([pts]),
                              True, (255, 0, 0), 2)
        cv2.putText(img, i[0], (x, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 20, 147), 2)
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
print(ROI_tup)
# for name,coords in ROI_tup:
#     print(f'{name} --- {coords}  {type(coords)} {coords.shape}')
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# # closing all open windows
cv2.destroyAllWindows()


