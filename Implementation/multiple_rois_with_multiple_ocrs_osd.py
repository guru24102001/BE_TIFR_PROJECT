import numpy as np
import cv2
import math
from scipy import ndimage
import pytesseract
from pytesseract import Output
import imutils
pytesseract.pytesseract.tesseract_cmd = 'D:\\roi_selection_and_ocr_with_orientation_correction-master\\tesseract.exe'

#####################################################################################################
# ORIENTATION CORRECTION/ADJUSTMENT


def orientation_correction(image_folder, save_image=True):
    paths = image_folder.readlines()
    orient_rotate_no = 0
    rotatedl = []
    for path in paths:
        IMAGE_FILE_LOCATION = path.strip()
        # print(IMAGE_FILE_LOCATION)
        img = cv2.imread(IMAGE_FILE_LOCATION)
        # GrayScale Conversion for the Canny Algorithm
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Canny Algorithm for edge detection was developed by John F. Canny not Kennedy!! :)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        # Using Houghlines to detect lines
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                                100, minLineLength=100, maxLineGap=5)
        print(lines)
        print(str(orient_rotate_no)+"*****************\n")
        if lines is None:
            orient_rotate_no += 1
            rotatedl.append('rotated/orientation_corrected' +
                            str(orient_rotate_no)+'.jpg')
            cv2.imwrite('rotated/orientation_corrected' +
                        str(orient_rotate_no)+'.jpg', img)
            continue
        # Finding angle of lines in polar coordinates
        angles = []
        for x1, y1, x2, y2 in lines[0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        # Getting the median angle
        median_angle = np.median(angles)

        # Rotating the image with this median angle
        img_rotated = ndimage.rotate(img, median_angle)
        orient_rotate_no += 1
        rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        rotated = imutils.rotate_bound(img_rotated, angle=results["rotate"])
        if save_image:
            rotatedl.append('rotated/orientation_corrected' +
                            str(orient_rotate_no)+'.jpg')
            cv2.imwrite('rotated/orientation_corrected' +
                        str(orient_rotate_no)+'.jpg', rotated)
    return rotatedl
#####################################################################################################
# OPTICAL CHARACTER RECOGNITION (OCR) ON ROI


def OCR(imgs):
    for img in imgs:
        text = pytesseract.image_to_string(img)
        print(f"The text in the selected region is as follows:\n{text}")
#####################################################################################################


def showROIS(ROIs, img_raw):
    # counter to save image with different name
    crop_number = 0
    imgs = []
    # loop over every bounding box save in array "ROIs"
    for rect in ROIs:
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        # crop roi from original image
        img_crop = img_raw[y1:y1+y2, x1:x1+x2]
        # show cropped image
        cv2.imshow("crop"+str(crop_number)+"Press Esc to exit", img_crop)
        # save cropped image
        imgs.append("crop"+str(crop_number)+".jpeg")
        cv2.imwrite("crop"+str(crop_number)+".jpeg", img_crop)
        crop_number += 1
        # hold window
        cv2.waitKey(0)
        OCR(imgs)
    # closing all open windows
    cv2.destroyAllWindows()

####################################################################################################

# REGION OF INTEREST (ROI) SELECTION


def createROIS(image_folder, mode):
    if mode == 1:
        for path in image_folder:
            # print('...................,,,,,,,')
            # read image
            img_raw = cv2.imread(path)
            # select ROIs function
            ROIs = cv2.selectROIs(
                "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img_raw)
            # print rectangle points of selected roi
            print(ROIs)
            showROIS(ROIs, img_raw)
    elif mode == 2:
        # print('//////////////////111111111////////')
        path = image_folder[0]
        img_raw = cv2.imread(path)
        # select ROIs function
        ROIs = cv2.selectROIs(
            "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img_raw)
        # print rectangle points of selected roi
        print(ROIs)
        showROIS(ROIs, img_raw)
        del image_folder[0]
        # print('//////////////2222222222222////////////')
        for path in image_folder:
            img_raw = cv2.imread(path)
            showROIS(ROIs, img_raw)
    else:
        for path in image_folder:
            print(pytesseract.image_to_string(path))


#####################################################################################################
if __name__ == '__main__':
    image_folder = open("images.txt", 'r')
    img_rotated = orientation_correction(image_folder, save_image=True)
    while True:
        num = input("a.Press 0 for OCR on complete document\nb.Press 1 for OCR using ROI on all images\nc.Press 2 for OCR using ROI template\nd.Press any other key for exit:\nSelect OCR mode:")
        if num not in ['0', '1', '2']:
            break
        createROIS(img_rotated, mode=int(num))
