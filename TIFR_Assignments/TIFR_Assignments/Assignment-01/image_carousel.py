from PyQt5.QtWidgets import QLabel, QApplication, QPushButton, QMainWindow, QFileDialog
import sys
import os
import cv2
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("carousel-window-updated.ui", self)

        
        #TO get the index of the next image in the list
        self.index = 0
        #Calculating the number of images To loop through the images once we reach the end
        self.image_count = 0
        #Folder location
        self.location = ""
        #To store all the files in the selected directory
        self.files = []

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        #Next button
        self.nextButton = self.findChild(QPushButton, "next")
        self.prevButton = self.findChild(QPushButton, "prev")
        self.open = self.findChild(QPushButton, "open")
        self.label = self.findChild(QLabel, "label")

        self.open.clicked.connect(self.openDirectory)
        self.nextButton.clicked.connect(self.nextImage)
        self.prevButton.clicked.connect(self.prevImage)

        self.show()


    def openDirectory(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Directory")
        # print(files)
        self.files = os.listdir(folder)
        VALID_FORMATS = ('.BMP', '.GIF', '.JPG', '.JPEG', '.PNG', '.PBM', '.PGM', '.PPM', '.TIFF', '.XBM')  # Image formats supported by Qt
        
        #Counting the number of valid image files in the folder so that it goes on in a cycle of showing images
        for file in self.files:
            if file.upper().endswith(VALID_FORMATS):
                self.image_count += 1
            else:
                self.files.remove(file)
        #Reducing one count because index starts from 0
        self.image_count = self.image_count - 1

        if (self.image_count == 0):
            print("THERE ARE NO IMAGES IN THIS FOLDER!")
            
        
        self.location = folder+"/"
        self.showImage(self.location + self.files[self.index])

    def nextImage(self):
        # print("Next Image clicked")
        #Whenever next button clicked the index is incremented and new image is shown
        self.index = (self.index % self.image_count) + 1
        self.showImage(self.location + self.files[self.index])

    def prevImage(self):
        # print("Prev imag clicked")
        #Whenever prev button clicked the index is decremented and new image is shown
        self.index = self.index % self.image_count - 1
        self.showImage(self.location + self.files[self.index])

    #A function which will map the image to QLAbel
    def showImage(self, path):
        img = cv2.imread(path)
        # img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.4, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)   
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # print(img.shape)
        # print(gray.shape)
        height, width, channel = img.shape
        step = channel*width
        qImg = QImage(img, width, height, step, QImage.Format_RGB888).rgbSwapped()

        print(qImg)

        self.pixmap = QPixmap.fromImage(qImg)
        #Scaling the width according to the window size
        self.pixmap = self.pixmap.scaledToWidth(510)
        self.label.setPixmap(self.pixmap)



app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()