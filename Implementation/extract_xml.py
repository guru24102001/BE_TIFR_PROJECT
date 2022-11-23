import xml.etree.ElementTree as ET
import os


FRAMES_DIRECTORY_PATH = os.path.join(os.path.dirname(__file__), './Data/frames_ground_truth')
FRAMES_GT_DIRECTORY_PATH = os.path.join(os.path.dirname(__file__), './Data/frames_ground_truth/test_dataset')

files = os.listdir(FRAMES_GT_DIRECTORY_PATH)

def extract_data(path, filename):    
    xml_file = open(os.path.join(FRAMES_GT_DIRECTORY_PATH, path), 'r').read()
    data = ET.fromstring(xml_file)
    objects = data.findall('object')
    
    bbox(objects, filename)
    
    
def bbox(objects, filename):
    for object in objects:
        name = object.find('name').text
        xmin = object.find('bndbox/xmin').text
        ymin = object.find('bndbox/ymin').text
        xmax = object.find('bndbox/xmax').text
        ymax = object.find('bndbox/ymax').text

        with open(os.path.join(FRAMES_GT_DIRECTORY_PATH, filename), 'a') as f:
            f.write(f"{xmin} {ymin} {xmax} {ymax} \"{name}\" \n")    
        print("Extracted text from the xml file ")
        print("================================================") 

for file in files:
    filename = file.split(".")[0]
    filename = f"gt_{filename}.txt"

    extract_data(file, filename)

