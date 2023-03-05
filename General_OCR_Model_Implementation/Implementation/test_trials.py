# import cv2
# image_folder=open("images.txt",'r')
# paths=image_folder.readlines()
# for path in paths:
#     IMAGE_FILE_LOCATION=path.strip()
#     # print(IMAGE_FILE_LOCATION)
#     input_img = cv2.imread(IMAGE_FILE_LOCATION)
#     cv2.imshow("images",input_img)
#     cv2.waitKey(0)
# import pytesseract
# from pytesseract import Output
# import cv2
# import imutils
# pytesseract.pytesseract.tesseract_cmd = 'D:\\roi_selection_and_ocr_with_orientation_correction-master\\tesseract.exe'
# image = cv2.imread("orientation_corrected1.jpg")
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
# rotated = imutils.rotate_bound(image, angle=results["rotate"])
# cv2.imshow("Rotated",rotated)
# cv2.waitKey(0)
# print(pytesseract.image_to_string(rotated))

