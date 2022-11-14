import easyocr
import time
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import shutil
import os
dataset = "https://github.com/SachaIZADI/Seven-Segment-OCR/tree/master/Datasets_Eleven/0"

start_time = time.time()

pd.set_option('display.max_rows', None)


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
            if np.mean(frame) < 25:
                print("Removed the frame")
                continue
            if frame_counter % 400 == 0:
                reduced_frames += 1
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
    file.write(fm+'\n')
    cv2.imwrite(fm, image_inverse)
    return


def enhance_frames(path, opt, file):
    dirs = os.listdir(path)
    if (opt == 0):
        directory = f'./enhanced_images/power_law/Stable_Video_1.mp4'
        if (os.path.isdir(directory)):
            shutil.rmtree(directory)

        os.makedirs(directory)
        for file_name in dirs:
            enhance_powerlaw(path + '/' + file_name, file_name, 1.6)
    else:
        directory = f'./enhanced_images/inverse_law/Stable_Video_1.mp4'
        if (os.path.isdir(directory)):
            shutil.rmtree(directory)

        os.makedirs(directory)
        for file_name in dirs:
            enhance_inverse(path + '/' + file_name, file_name, file)
            # enhance_powerlaw(path + '/enhanced_images/inverse_law/Stable_Video_1.mp4/' + file_name, file_name, 1.6)


#####################################################################################################
# OPTICAL CHARACTER RECOGNITION (OCR) ON ROI
d = {}
reader = easyocr.Reader(['en'])


def OCR(imgs, mode):
    if mode == 2:
        for img in imgs:
            image = img[1]
            # print(images)
            result = reader.readtext(image)
            # print(result)
            if img[0] not in d:
                d[img[0]] = []
                if (len(result) > 0 and len(result[0]) >= 2):
                    d[img[0]].append(result[0][-2])
                else:
                    # continue
                    d[img[0]].append(None)
            else:
                if (len(result) > 0 and len(result[0]) >= 2):
                    d[img[0]].append(result[0][-2])
                else:
                    # continue
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
# [139, 179, 101, 117]
# [469, 495, 101, 117]
def showROIS(ROIs, img_raw,fp):
    # counter to save image with different name
    crop_number = 0
    imgs = []
    imgp=Image.open(fp)
    # loop over every bounding box save in array "ROIs"
    for rect in ROIs:
        if (isinstance(rect[1][0], np.integer)):
            x1 = rect[1][0]  # tlx
            y1 = rect[1][1]  # brx
            x2 = rect[1][2]  # bry
            y2 = rect[1][3]  # tly
            # crop roi from original image
            if (rect[0].startswith("roi")):
                # img_crop = img_raw[y1:y1+y2, x1:x1+x2]
                img_res = imgp.crop((x1, x2, y1, y2))
                imgs.append((rect[0], "crop"+str(crop_number)+".jpeg"))
                img_res.save("crop"+str(crop_number)+".jpeg")
                crop_number += 1
            else:
                img_crop = img_raw[y1:y1+y2, x1:x1+x2]
                # show cropped image
                # save cropped image
                imgs.append((rect[0], "crop"+str(crop_number)+".jpeg"))
                cv2.imwrite("crop"+str(crop_number)+".jpeg", img_crop)
                crop_number += 1
            # hold window
            # cv2.waitKey(0)
        else:
            coords = [tuple(l) for l in rect[1]]
            pts = np.array(coords, dtype="float32")
            warped = four_point_transform(img_raw, pts)
            imgs.append((rect[0], "crop"+str(crop_number)+".jpeg"))
            cv2.imwrite("crop"+str(crop_number)+".jpeg", warped)
            crop_number += 1

    OCR(imgs, 2)
    # closing all open windows
    cv2.destroyAllWindows()
####################################################################################################

# REGION OF INTEREST (ROI) SELECTIO

reader = easyocr.Reader(['en'])
def createROIS(image_folder, mode):
    files = os.listdir(image_folder)
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
        print(p)
        img = cv2.imread(p)
        img_raw = cv2.imread(p)
        # image = keras_ocr.tools.read(p)
        img_det = reader.detect(img)
        ctr = 0
        img_tup = []
        for i in img_det:
            for j in i[0]:
                if (isinstance(j[0], np.integer)):
                    print(j)
                    ctr += 1
                    st = 'roi'+str(ctr)
                    img_tup.append((st, np.array(j)))
                if (isinstance(j[0], list)):
                    print(j)
                    ctr += 1
                    st = 'roi'+str(ctr)
                    img_tup.append((st, j))
        for i in img_tup:
            if (isinstance(i[1][0], np.integer)):
                x1 = int(i[1][0])
                y1 = int(i[1][3])
                x2 = int(i[1][1])
                y2 = int(i[1][2])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, i[0], (x1, y1+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 49, 49), 2)
            else:
                pts = i[1]
                pts = np.array(pts)
                x = int(pts[0][0])
                y = int(pts[0][1])
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, np.int32([pts]),
                                    True, (255, 0, 0), 2)
                cv2.putText(img, i[0], (x, y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 20, 147), 2)
        cv2.imshow('window_name', img)
        cv2.waitKey(0)
        r = input("Please select the ROIs on which you want OCR to be done by entering them with comma as a separator:")
        rl = r.split(",")
        rm = list(map(lambda x: "roi"+x, rl))
        rf = tuple(filter(lambda x: x[0] in rm, img_tup))
        print(rf)
        ROIs = cv2.selectROIs(
            "Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img)
        ROI_tup = []
        for i in ROIs:
            name = input(
                'Please enter the attribute name for the selected ROIS:')
            ROI_tup.append((name, i))
        ROI_tup.extend(rf)
        print(ROI_tup)
        # for name, coords in ROI_tup:
        #     print(f'{name} --- {coords}  {type(coords)} {coords.shape}')

        showROIS(ROI_tup, img_raw, image_folder + "/" + path)
        del files[0]
        for path in files:
            img_raw = cv2.imread(image_folder + "/" + path)
            showROIS(ROI_tup, img_raw, image_folder + "/" + path)
    else:
        for path in files:
            print(image_folder + "/" + path)

            abcd = reader.readtext(image_folder + "/" + path)
            for line in abcd:
                print(f"{line[0]}      {line[1]}")
            print("*****************************************************************")

# [[391, 25], [487, 25], [487, 43], [391, 43]]      DSO-X 3014A
# [[390, 25], [487, 25], [487, 46], [390, 46]]      DSO-X 3014A
# [[391, 27], [487, 27], [487, 45], [391, 45]]      DSO-X 3014A
# [[390, 25], [487, 25], [487, 46], [390, 46]]      DSO-X 3014A

#####################################################################################################


file = open('images.txt', 'w')
# create_frames("./dataset/Stable_Video_1.mp4", 1, file)
create_frames("./dataset/Stable_Video_1.mp4")
enhance_frames("./images/Stable_Video_1.mp4", 1, file)
# image_folder = open("./images.txt", 'r')

createROIS("./enhanced_images/inverse_law/Stable_Video_1.mp4", mode=2)
formated_output = pd.DataFrame(d)

# formated_output.style.set_table_styles([{'selector' : '',
# 'props' : [('border',
# '2px solid white')]}])

print(formated_output)
file.close()
print("Execution time: ", time.time() - start_time)


# print(image_folder[0])
# paths = image_folder.readlines()
# print("***********************************")
# print(paths)
# rotatedl = []
# for path in paths:
#     rotatedl.append(path.strip())
# print(rotatedl)
