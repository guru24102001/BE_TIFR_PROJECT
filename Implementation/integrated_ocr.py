dataset = "https://github.com/SachaIZADI/Seven-Segment-OCR/tree/master/Datasets_Eleven/0"
import os
import shutil
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time
import easyocr

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
                cv2.imwrite(f'./images/{name}/frame_{frame_counter}.jpg', resized_frame)
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
    frame_normed = 255 * (result - result.min()) / (result.max() - result.min())
    frame_normed = np.array(frame_normed, np.int64)

    cv2.imwrite(f'./enhanced_images/Stable_Video_1.mp4/{file_name}.jpg', frame_normed)
    return

def enhance_inverse(path, file_name, file):
    image = Image.open(path)
    image_array = np.asarray(image)
    image_inverse = 255 - image_array
    fm = './enhanced_images/inverse_law/Stable_Video_1.mp4/'+file_name
    file.write(fm+'\n')
    cv2.imwrite(fm, image_inverse)
    return

def enhance_frames(path, opt,file):
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
def OCR(imgs,mode):
    if mode==2:
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


def showROIS(ROIs, img_raw,names):
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
        # save cropped image
        imgs.append((names[crop_number], "crop"+str(crop_number)+".jpeg"))
        cv2.imwrite("crop"+str(crop_number)+".jpeg", img_crop)
        crop_number += 1
        # hold window
        cv2.waitKey(0)
    OCR(imgs,2)
    # closing all open windows
    cv2.destroyAllWindows()
####################################################################################################

# REGION OF INTEREST (ROI) SELECTIO

def createROIS(image_folder, mode):
    files = os.listdir(image_folder)
    if mode == 1:
        for path in image_folder:
            img_raw = cv2.imread(image_folder + "/" + path)
            # select ROIs function
            ROIs = cv2.selectROIs("Select ROIs and press Enter to move further, Press Esc to exit, Press c to clear selection", img_raw)
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
        names=[]

        for i in ROIs:
            name = input(
                'Please enter the attribute name for the selected ROIS:')
            names.append(name)

        showROIS(ROIs, img_raw, names)
        del files[0]
        for path in files:
            img_raw = cv2.imread(image_folder + "/" + path)
            showROIS(ROIs, img_raw, names)
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
formated_output = pd.DataFrame(d);

# formated_output.style.set_table_styles([{'selector' : '',
# 'props' : [('border',
# '2px solid white')]}])

print(formated_output)
file.close()
print("Execution time: ",time.time() - start_time)


# print(image_folder[0])
# paths = image_folder.readlines()
# print("***********************************")
# print(paths)
# rotatedl = []
# for path in paths:
#     rotatedl.append(path.strip())
# print(rotatedl)
