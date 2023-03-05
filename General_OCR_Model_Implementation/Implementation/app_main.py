import easyocr
import time
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import shutil
import os
import tensorflow as tf
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import time
import keras_ocr
start_time = time.time()
assert tf.test.is_gpu_available(), 'No GPU is available.'

#####################################################################################################
# FRAME CREATION AND THEIR EHANCEMENTS


def create_frames(path):
    """Creates frames from the given input video
    Args:path of the video"""
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
                reduced_frames += 1
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
        f'./enhanced_images/{dataset}/{file_name}.jpg', frame_normed)
    return


def enhance_inverse(path, file_name, file):
    image = Image.open(path)
    image_array = np.asarray(image)
    image_inverse = 255 - image_array
    fm = './enhanced_images/inverse_law/'+dataset+'/'+file_name
    # print(fm)
    file.write(fm+'\n')
    cv2.imwrite(fm, image_inverse)
    return


def enhance_frames(path, opt, file):
    dirs = os.listdir(path)
    # print("runnnnel")
    if (opt == 0):
        # Applying Power Law
        directory = f'./enhanced_images/power_law/'+dataset
        if (os.path.isdir(directory)):
            shutil.rmtree(directory)

        os.makedirs(directory)
        for file_name in dirs:
            enhance_powerlaw(path + '/' + file_name, file_name, 1.6)
    else:
        # Applying Inverse law
        directory = f'./enhanced_images/inverse_law/'+dataset
        if (os.path.isdir(directory)):
            shutil.rmtree(directory)

        os.makedirs(directory)
        for file_name in dirs:
            enhance_inverse(path + '/' + file_name, file_name, file)


#####################################################################################################
# OPTICAL CHARACTER RECOGNITION (OCR) ON ROI
d = {}
reader = easyocr.Reader(['en'], recog_network='iter_30000', gpu=True)


def OCR(imgs, mode):
    if mode == 2:
        for img in imgs:
            image = img[1]
            result = reader.readtext(image)
            if img[0] not in d:
                d[img[0]] = []
                if (len(result) > 0 and len(result[0]) >= 2):
                    d[img[0]].append(result[0][-2])
                else:
                    d[img[0]].append(None)
            else:
                if (len(result) > 0 and len(result[0]) >= 2):
                    d[img[0]].append(result[0][-2])
                else:
                    d[img[0]].append(None)


#####################################################################################################
####################################################################################################
# for kocr_detection_orientation_check

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

#####################################################################################################


def showROIS(ROIs, img_raw):
    # counter to save image with different name
    crop_number = 0
    imgs = []
    # loop over every bounding box save in array "ROIs"
    for rect in ROIs:
        if rect[1].shape == (4,):
            x1 = rect[1][0]
            y1 = rect[1][1]
            x2 = rect[1][2]
            y2 = rect[1][3]
            # crop roi from original image
            img_crop = img_raw[y1:y1+y2, x1:x1+x2]
            # save cropped image
            imgs.append((rect[0], "crop"+str(crop_number)+".jpeg"))
            cv2.imwrite("crop"+str(crop_number)+".jpeg", img_crop)
            crop_number += 1
            # hold window
            cv2.waitKey(0)
        else:
            warped = four_point_transform(img_raw, rect[1])
            imgs.append((rect[0], "crop"+str(crop_number)+".jpeg"))
            cv2.imwrite("crop"+str(crop_number)+".jpeg", warped)
            crop_number += 1

    OCR(imgs, 2)
    # closing all open windows
    cv2.destroyAllWindows()
####################################################################################################
# REGION OF INTEREST (ROI) SELECTION


det = keras_ocr.detection.Detector()
det.model.load_weights('./detector_icdar2013.h5')


def createROIS(image_folder, mode):
    files = os.listdir(image_folder)
    print(files)
    lst = []
    for file in files:
        lst.append(image_folder + "/" + file)
    print(lst)

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
        p = image_folder + "/" + path
        img_raw = cv2.imread(p)
        image = keras_ocr.tools.read(p)
        img_det = det.detect([image])
        img_tup = []
        for i in range(len(img_det[0])):
            st = 'roi'+str(i+1)
            img_tup.append((st, img_det[0][i]))
        img = keras_ocr.tools.drawBoxes(image, img_det[0], color=(
            36, 255, 12), thickness=1, boxes_format='boxes')
        for i in img_tup:
            x = int(i[1][0][0])
            y = int(i[1][0][1])
            cv2.putText(img, i[0], (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 20, 147), 2)

        cv2.imshow('window_name', img)
        cv2.waitKey(0)
        r = input(
            "Please select the ROIs on which you want OCR to be done by entering them with comma as a separator:")
        rl = r.split(",")
        rm = list(map(lambda x: "roi"+x, rl))
        rf = tuple(filter(lambda x: x[0] in rm, img_tup))
        print(rf)
        ROIs = cv2.selectROIs(
            "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img)
        # print rectangle points of selected roi
        print(ROIs)
        ROI_tup = []
        for i in ROIs:
            name = input(
                'Please enter the attribute name for the selected ROIS:')
            ROI_tup.append((name, i))
        ROI_tup.extend(rf)
        for name, coords in ROI_tup:
            print(f'{name} --- {coords}  {type(coords)} {coords.shape}')
        showROIS(ROI_tup, img_raw)
        del files[0]
        for path in files:
            img_raw = cv2.imread(image_folder + "/" + path)
            showROIS(ROI_tup, img_raw)
    else:
        for path in files:
            print(image_folder + "/" + path)

            abcd = reader.readtext(image_folder + "/" + path)
            for line in abcd:
                print(f"{line[0]}      {line[1]}")
            print("*****************************************************************")

#####################################################################################################
# MAIN DRIVER FUNCTIONS


file = open('images.txt', 'w')
dataset = "Stable_Video_1.mp4"
create_frames("./dataset/"+dataset)
enhance_frames("./images/"+dataset, 1, file)
createROIS("./enhanced_images/inverse_law/"+dataset, mode=2)
formated_output = pd.DataFrame(d)
print(formated_output)
file.close()
print("Execution time: ", time.time() - start_time)
