import torch
import os
import pandas as pd
import csv

# Model Load

model = torch.hub.load('ultralytics/yolov5', 'custom', '../V2_YOLOv5Character-20230224T134754Z-001/YOLOv5Character/yolov5/runs/train/exp3/weights/best.pt')  # custom trained model

# Images Path


# im = 'Test_Images_And_Actual_Annotations/data_2251.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

images = 'D:/Akaash/VESIT_BE_PROJECT/YOLO_v5_Character_Updated/V2_Local_Support_Files/'

for im in os.listdir(images):

    # Pararmeter Tweaking
    # Change confidence score from 0.25 (default)
    if (im.endswith(".jpg")):

        model.conf=0.50
        model.iou=0.05
        model.multi_label = False
        model.max_det = 18


        # Inference
        results = model(im, size = 960)

        # Results
        # results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

        #Sort predicted bounding boxes based on the values of x coordinate of the boxes
        output_table = results.pandas().xyxy[0].sort_values('xmin')  # im predictions (pandas)

        print(output_table)

        # To write the test image predicted output in the .csv file
        test_output = pd.DataFrame(output_table)
        test_output.to_csv('Test_Output_Predicted_Annotations/' + 'output.csv')

        # To read the created .csv file
        test_output_read = pd.read_csv('Test_Output_Predicted_Annotations/' + 'output.csv')

        # total no. of records(each record corresponds to one digits) in the file
        size_of_table = test_output_read['class'].size
        print("Total number of digits : ", size_of_table)


        # Logic Combine digits to create a single valued number
        result = ["", "", "", ""]
        c = 0
        digits = []

        index1=0
        index2=0
        index3=0
        # print("Done")

        # for i in range(size_of_table):
        #     print(test_output_read['xmin'][i], test_output_read['ymin'][i], test_output_read['xmax'][i], test_output_read['ymax'][i], test_output_read['confidence'][i], test_output_read['class'][i])
        for i in range(size_of_table-1):
            digits.append(test_output_read['class'][i])

        # print("Done")
        
        max_dist = 0
        for i in range(size_of_table-1):
            if(test_output_read['xmin'][i+1] - test_output_read['xmax'][i] > max_dist):
                max_dist = test_output_read['xmin'][i+1] - test_output_read['xmax'][i]
                index2 = i;
        # print("Done")
        
        i=0
        while(i < index2):
            if(test_output_read['xmin'][index2] - test_output_read['xmax'][i] > 0.34*max_dist):
                index1 = i;
            i = i+1
        # print("Done")
        
        i = size_of_table-1
        while (i > index2):
            if(test_output_read['xmin'][i] - test_output_read['xmax'][index2] > 1.125*max_dist):
                index3 = i;
            i = i-1
        # print("Done")
        
        i = 0
        c = 0
        while(i<=index1):
            result[c] = result[c] + str(test_output_read['class'][i])
            i = i+1
        # print("Done")

        c = 1
        while(i<=index2):
            result[c] = result[c] + str(test_output_read['class'][i])
            i = i+1
        # print("Done")

        c = 2
        while(i<=index3):
            result[c] = result[c] + str(test_output_read['class'][i])
            i = i+1
        # print("Done")

        c = 3
        while(i<size_of_table):
            result[c] = result[c] + str(test_output_read['class'][i])
            i = i+1

        # print("Done")
        # print(index1)
        # print(index2)
        # print(index3)


        print("First Value : ", result[0]);
        print("Second Value : ", result[1]);
        print("Third Value : ", result[2]);
        print("Fourth Value : ", result[3]);
        print("Done")
                # print(test_output_read['xmin'][i], test_output_read['ymin'][i], test_output_read['xmax'][i], test_output_read['ymax'][i]) 



        # CSV sheet which contains all the outputs from the testing set.

        field_names = ['Display_1', 'Display_2', 'Display_3', 'Display_4']
        dict = {"Display_1":result[0], "Display_2":result[1], "Display_3":result[2], "Display_4":result[3]}
        with open('Test_Output_Predicted_Annotations/Predicted_Values.csv', 'a') as csv_file:
            dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
        
            dict_object.writerow(dict)