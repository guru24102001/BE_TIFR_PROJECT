import cv2
import time
import numpy as np
import pandas as pd
import keras_ocr
import easyocr
import os
import thread_queue
import tensorflow as tf
assert tf.config.list_physical_devices('GPU'), 'No GPU is available.'
import datetime
t = time.time()
#####################################################################################################
# OPTICAL CHARACTER RECOGNITION (OCR) ON ROI
d = {}
reader = easyocr.Reader(['en'], gpu=True)



def OCR(imgs):
    ts = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
    print(f"OCR at {ts}")
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
    d['Time']=ts
    df=pd.DataFrame(d)
    df.to_csv('values.csv',index=False)


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
    print("Show ROI")
    crop_number = 0
    imgs = []
    # loop over every bounding box save in array "ROIs"
    for rect in ROIs:
        # print("In loop")
        if rect[1].shape == (4,):
            # print("4-point")
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
            # cv2.waitKey(0)
        else:
            # print("warped-point")

            warped = four_point_transform(img_raw, rect[1])
            imgs.append((rect[0], "crop"+str(crop_number)+".jpeg"))
            cv2.imwrite("crop"+str(crop_number)+".jpeg", warped)
            crop_number += 1

    OCR(imgs)
    # closing all open windows
    cv2.destroyAllWindows()


####################################################################################################
############################################################################################################
# Detecting and Creating ROIs
det = keras_ocr.detection.Detector()
det.model.load_weights('./detector_icdar2013.h5')


def createROIs(filename):
    print("Detection started to create ROIs")
    # img_raw = cv2.imread(filename)
    image = keras_ocr.tools.read(filename)
    img_det = det.detect([image])
    print(img_det)
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

    cv2.imshow('Detected ROIs , Press Esc to exit', img)
    cv2.waitKey(0)
    r = input(
        "Please select the ROIs on which you want OCR to be done by entering them with comma as a separator:")
    rl = r.split(",")
    rm = list(map(lambda x: "roi"+x, rl))
    rf = tuple(filter(lambda x: x[0] in rm, img_tup))
    print(rf)
    cv2.destroyAllWindows()
    ROIs = cv2.selectROIs(
        "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img)
    # print rectangle points of selected roi
    print(ROIs)
    cv2.destroyAllWindows()
    ROI_tup = []
    for i in ROIs:
        name = input(
            'Please enter the attribute name for the selected ROIS:')
        ROI_tup.append((name, i))
    ROI_tup.extend(rf)
    for name, coords in ROI_tup:
        print(f'{name} --- {coords}  {type(coords)} {coords.shape}')
    cv2.destroyAllWindows()
    return ROI_tup
    # showROIS(ROI_tup, img_raw)
    # for path in files:
    #     img_raw = cv2.imread(image_folder + "/" + path)
    #     showROIS(ROI_tup, img_raw)


############################################################################################################
print('Starting and setting up GPU')
# if capture.isOpened() is False:
#   print("[Exiting]: Error accessing webcam stream.")
#   exit(0)
# fps_input_stream = int(capture.get(5))  # get fps of the hardware
# print(f"FPS of input stream is {fps_input_stream}")

print('Camera check')
############################################################################################################
# capture = cv2.VideoCapture(
#     "dataset/Stable_Video_1.mp4")
webcam_stream = thread_queue.WebcamStream(
    stream_id='rtsp://ajay:aju13@192.168.0.102:8080/h264_pcm.sdp')
# webcam_stream = threading_test.WebcamStream(
#     stream_id='dataset\Stable_Video_1.mp4')
# capture = cv2.VideoCapture(
#     "rtsp://192.168.0.101:8080/h264_pcm.sdp")
webcam_stream.start()
############################################################################################################
while True:
    frame = webcam_stream.read()

    height, width, layers = frame.shape
    new_h = height / 2
    new_w = width / 2
    frame = cv2.resize(frame, (int(new_w), int(new_h)))
    cv2.imwrite('det_roi_file/roi.jpg', frame)
    cv2.imshow('Text Detection frame, Press q to select', frame)
    # cv2.waitKey(0)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
tup_ROI = createROIs('det_roi_file/roi.jpg')

avg = 0
counter = 0
start_time = time.time()
print('OCR starts')
try:
    while (True):
        if webcam_stream.stopped is True:
            break
        else:
            frame = webcam_stream.read()
            height, width, layers = frame.shape
            new_h = height / 2
            new_w = width / 2
            frame = cv2.resize(frame, (int(new_w), int(new_h)))
            print(f' Frames came till now {counter}')
            cv2.imwrite("frame_folder/frame"+str(counter)+".jpg", frame)
            img_raw = cv2.imread("frame_folder/frame"+str(counter)+".jpg")
            ocr_stime = time.time()
            showROIS(tup_ROI, img_raw)
            os.remove("frame_folder/frame"+str(counter)+".jpg")
            ocr_etime = time.time()
            print(d)
            telaps = ocr_etime-ocr_stime
            avg = (counter*avg+telaps)/(counter+1)
            print(f"Time taken for OCR on a frame: {telaps} {avg}")
            counter += 1
        if cv2.waitKey(1) == ord("q"):
            break
except KeyboardInterrupt:
    print("Program Stopped!!")
print(f'Average time taken: {avg}')
webcam_stream.stop()
cv2.destroyAllWindows()
print(d)
csvfnm = 'values'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
os.rename('values.csv',csvfnm)
print("--- %s seconds ---" % (time.time() - start_time))


# while (True):

#    ret, frame = capture.read()
#    if ret:
#     print("ran")
#     cv2.imwrite("frame_folder/frame%d.jpg" % counter, frame)
#     img_raw = cv2.imread()
#     counter += 1
#     if counter == 10:
#       counter = 0

#    cv2.imshow('livestream', frame)
#    if cv2.waitKey(1) == ord("q"):
#       break
