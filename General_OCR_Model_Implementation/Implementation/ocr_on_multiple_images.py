import pytesseract
from multiple_rois_with_multiple_ocrs_osd import createROIS
pytesseract.pytesseract.tesseract_cmd = 'D:\\roi_selection_and_ocr_with_orientation_correction-master\\tesseract.exe'
file=open('images.txt','r')
img_folder=[]
for line in file.readlines():
    img_folder.append(line.strip())
print(img_folder)
while True:
    num = input("a.Press 0 for OCR on complete document\nb.Press 1 for OCR using ROI on all images\nc.Press 2 for OCR using ROI template\nd.Press any other key for exit:\nSelect OCR mode:")
    if num not in ['0', '1', '2']:
        break
    createROIS(img_folder, mode=int(num))
