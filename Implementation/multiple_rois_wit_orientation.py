######################################################################################################
# ABOUT THE FILE:
# This is roi_selection_and_ocr.py file includes:
# 1) Automatic orientation adjustment
# 2) User Interface for the user to select the Region of Interest (ROI or roi) on the image.
# 3) Extraction of the selected ROI.
# 4) Optical Character Recognition (OCR) on the selected ROI
# 5) Display of the output text in the terminal window
######################################################################################################
######################################################################################################
# NOTE TO USER/READER:
# 1) This module is developed and tested using Python 3.6 and above and the author
#    recommends the same for better and smooth execution.
# 2) Kindly install necessary libraries : numpy, OpenCV, scipy, pytesseract
# 3) Ensure the necessary environment variables are set correctly.
# 3) For demonstration purposes, the user/reader is adviced to mention the location of the input image
#    in the IMAGE_FILE_LOCATION variable before executing.
######################################################################################################

import numpy as np
import cv2
import math
from scipy import ndimage
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'D:\\roi_selection_and_ocr_with_orientation_correction-master\\tesseract.exe'
IMAGE_FILE_LOCATION = "dataset\\1.png"  # Photo by Amanda Jones on Unsplash
input_img = cv2.imread(IMAGE_FILE_LOCATION)  # image read

#####################################################################################################
# ORIENTATION CORRECTION/ADJUSTMENT


def orientation_correction(img, save_image=False):
    # GrayScale Conversion for the Canny Algorithm
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny Algorithm for edge detection was developed by John F. Canny not Kennedy!! :)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    # Using Houghlines to detect lines
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)

    # Finding angle of lines in polar coordinates
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Getting the median angle
    median_angle = np.median(angles)

    # Rotating the image with this median angle
    img_rotated = ndimage.rotate(img, median_angle)

    if save_image:
        cv2.imwrite('orientation_corrected.jpg', img_rotated)
    return img_rotated
#####################################################################################################


img_rotated = orientation_correction(input_img)
result = cv2.imwrite(r"img_rotated.jpeg", img_rotated) 
if result == True: 
    print("File saved successfully") 
else: print("Error in saving file")
# cv2.imwrite(str(img_rotated)+".jpeg", img_rotated)
# print("ran")
# time.sleep(5)
#####################################################################################################
# REGION OF INTEREST (ROI) SELECTION

# read image
IMAGE_LOCATION="img_rotated.jpeg"
img_raw = cv2.imread(IMAGE_LOCATION)

# select ROIs function
ROIs = cv2.selectROIs("Select Rois", img_raw)

# print rectangle points of selected roi
print(ROIs)

# Crop selected roi ffrom raw image

# counter to save image with different name
crop_number = 0
imgs=[]
# loop over every bounding box save in array "ROIs" 
for rect in ROIs: 
    x1 = rect[0] 
    y1 = rect[1] 
    x2 = rect[2] 
    y2 = rect[3] 
    # crop roi from original image 
    img_crop = img_raw[y1:y1+y2, x1:x1+x2] 
    #show cropped image 
    cv2.imshow("crop"+str(crop_number),img_crop) 
    # save cropped image 
    imgs.append("crop"+str(crop_number)+".jpeg")
    cv2.imwrite("crop"+str(crop_number)+".jpeg",img_crop) 
    crop_number+=1 
    # hold window 
    cv2.waitKey(0)
  
# closing all open windows 
cv2.destroyAllWindows()  
    
#####################################################################################################

#####################################################################################################
# OPTICAL CHARACTER RECOGNITION (OCR) ON ROI
for img in imgs:
    text = pytesseract.image_to_string(img)
    print("The text in the selected region is as follows:")
    print(text)