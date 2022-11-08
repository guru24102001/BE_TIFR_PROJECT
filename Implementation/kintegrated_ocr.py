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
start_time = time.time()

# pytesseract.pytesseract.tesseract_cmd = 'D:\\roi_selection_and_ocr_with_orientation_correction-master\\tesseract.exe'


# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

def create_frames(path):
    frame_counter = 0
    reduced_frames = 0
    capture = cv2.VideoCapture(path)
    name = path.split("/")[-1]
    directory = f'./images/{name}'
    if (os.path.isdir(directory)):
        shutil.rmtree(directory)

    os.makedirs(directory)
    while True:
        success, frame = capture.read()
        if success:
            # Deleting low resolution frame
            if np.mean(frame) < 15:
                print("Removed the frame")
                continue
            if frame_counter % 400 == 0:
                resized_frame = cv2.resize(frame, (960, 540))
                cv2.imwrite(
                    f'./images/{name}/frame_{frame_counter}.jpg', resized_frame)
        else:
            break
        frame_counter += 1
    print(f'There were {frame_counter} frames in this video')
    print(f'After reducing frames there are {reduced_frames} in this video. ')
    capture.release()
    return


def enhance_powerlaw(path, file_name, gamma):
    image = cv2.imread(path)
    normalized_image = image / 255.0
    result = cv2.pow(normalized_image, gamma)
    frame_normed = 255 * (result - result.min()) / \
        (result.max() - result.min())
    frame_normed = np.array(frame_normed, np.int64)

    cv2.imwrite(
        f'./enhanced_images/Stable_Video_1.mp4/{file_name}.jpg', frame_normed)
    return


def enhance_inverse(path, file_name, file):
    image = Image.open(path)
    image_array = np.asarray(image)
    image_inverse = 255 - image_array
    fm = './enhanced_images/inverse_law/Stable_Video_1.mp4/'+file_name
    # print(fm)
    file.write(fm+'\n')
    cv2.imwrite(fm, image_inverse)
    return


def enhance_frames(path, opt, file):
    dirs = os.listdir(path)
    # print("runnnnel")
    if (opt == 0):
        # Applying Power Law
        directory = f'./enhanced_images/power_law/Stable_Video_1.mp4'
        if (os.path.isdir(directory)):
            shutil.rmtree(directory)

        os.makedirs(directory)
        for file_name in dirs:
            enhance_powerlaw(path + '/' + file_name, file_name, 1.6)
    else:
        # Applying Inverse law
        directory = f'./enhanced_images/inverse_law/Stable_Video_1.mp4'
        if (os.path.isdir(directory)):
            shutil.rmtree(directory)
        # print("runnnnel")

        os.makedirs(directory)
        for file_name in dirs:
            enhance_inverse(path + '/' + file_name, file_name, file)


# *********************************
# * PERFORMING OCR USING KERAS-OCR


#####################################################################################################
# OPTICAL CHARACTER RECOGNITION (OCR) ON ROI

d = {}
# keras-ocr
reg = keras_ocr.recognition.Recognizer()
# reader = easyocr.Reader(['en'])


def OCR(imgs, mode):
    if mode == 0:
        pass
        # for img in imgs:
        #     text = pytesseract.image_to_string(img[1])
        #     if img[0] not in d:
        #         d[img[0]] = []
        #         d[img[0]].append(text)
        #     else:
        #         d[img[0]].append(text)
        #     # print(f"OCR of {img[0]}:\n{text}")
    elif mode == 1:
        for img in imgs:
            images = img[1]
            # print(images)
            prediction_imgall = reg.recognize(images)
            # print(prediction_imgall)
            if img[0] not in d:
                d[img[0]] = []
                d[img[0]].append(prediction_imgall)
            else:
                d[img[0]].append(prediction_imgall)
        # print(f"OCR of {img[0]}:\n{prediction_imgall}")
    # elif mode==2:
    #     for img in imgs:
    #         image = img[1]
    #         # print(images)
    #         result = reader.readtext(image)
    #         print(result)
    #         if img[0] not in d:
    #             d[img[0]] = []
    #             d[img[0]].append(result[0][-2])
    #         else:
    #             d[img[0]].append(result[0][-2])
    #         print(f"OCR of {img[0]}:\n{result[0][-2]}")


#####################################################################################################


def showROIS(ROIs, img_raw, names):
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
        # cv2.imshow("crop"+str(crop_number)+name+"Press Esc to exit", img_crop)
        # save cropped image
        imgs.append((names[crop_number], "crop"+str(crop_number)+".jpeg"))
        cv2.imwrite("crop"+str(crop_number)+".jpeg", img_crop)
        crop_number += 1
        # hold window
        cv2.waitKey(0)
    OCR(imgs, 1)
    # closing all open windows
    cv2.destroyAllWindows()
####################################################################################################

# REGION OF INTEREST (ROI) SELECTION


def createROIS(image_folder, mode):
    files = os.listdir(image_folder)
    print(files)
    lst=[]
    for file in files:
        lst.append(image_folder + "/" + file)

    if mode == 1:
        for path in image_folder:
            img_raw = cv2.imread(image_folder + "/" + path)
            # select ROIs function
            ROIs = cv2.selectROIs(
                "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img_raw)
            # print rectangle points of selected roi
            print(ROIs)
            showROIS(ROIs, img_raw)
    elif mode == 2:
        path = files[0]
        img_raw = cv2.imread(image_folder + "/" + path)
        # select ROIs function
        ROIs = cv2.selectROIs(
            "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img_raw)
        # print rectangle points of selected roi
        print(ROIs)
        names = []

        for i in ROIs:
            name = input(
                'Please enter the attribute name for the selected ROIS:')
            names.append(name)

        showROIS(ROIs, img_raw, names)
        del files[0]
        # print('//////////////2222222222222////////////')
        for path in files:
            img_raw = cv2.imread(image_folder + "/" + path)
            showROIS(ROIs, img_raw, names)
    else:
        
        images = [
            keras_ocr.tools.read(url) for url in lst
        ]
        prediction_groups = pipeline.recognize(images)
        abcd = prediction_groups
        for lst in abcd:
            for tup in lst:
                arr = tup[1]
                print(f"{tup[0]}    {arr[0]}  {arr[1]}  {arr[2]}  {arr[3]}")


        # print(lst)
        # print(image_folder + "/" + path)
        # abcd = reg.recognize(image_folder + "/" + path)
        # print(abcd)
        # for line in abcd:
        #     print(f"{line[0]}      {line[1]}")
        print("*****************************************************************")


# dsox    [391.  28.]  [439.  28.]  [439.  43.]  [391.  43.]
# dsox    [391.  28.]  [439.  28.]  [439.  42.]  [391.  42.]
# dsox    [391.  28.]  [439.  28.]  [439.  43.]  [391.  43.]
# dsox    [391.  28.]  [439.  28.]  [439.  42.]  [391.  42.]


#####################################################################################################
file = open('images.txt', 'w')
# create_frames("./dataset/Stable_Video_1.mp4")
# enhance_frames("./images/Stable_Video_1.mp4", 1, file)
# image_folder = open("images.txt", 'r')
# paths = image_folder.readlines()
# rotatedl = []
# for path in paths:
#     rotatedl.append(path.strip())
# print(rotatedl)
# createROIS(rotatedl, mode=2)
# print(d)
# file.close()
# print("Execution time: ", time.time() - start_time)


create_frames("./dataset/Stable_Video_1.mp4")
enhance_frames("./images/Stable_Video_1.mp4", 1, file)
# image_folder = open("./images.txt", 'r')
createROIS("./enhanced_images/inverse_law/Stable_Video_1.mp4", mode=3)
# formated_output = pd.DataFrame(d);

# formated_output.style.set_table_styles([{'selector' : '',
# 'props' : [('border',
# '2px solid white')]}])

print(d)
# file.close()
print("Execution time: ",time.time() - start_time)