#!/usr/bin/python

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *

from threading import Thread

import numpy as np
import argparse
import imutils
import os
import glob
import cv2
import argparse

import DetectChars
import DetectPlates
import PossiblePlate

import serial
import requests
import time

# import _thread
import threading
from time import gmtime, strftime

class Window(QMainWindow):

    def __init__(self, template_dir, portUSB):

        # Inherit properies from QMainWindow
        super().__init__()

        # Setting geometry attributes of MainWindow
        self.left = 10
        self.top = 10
        self.width = 720
        self.height = 480
        self.project_num = 0
        self.project_name = "_"
        self.project_list = []
        self.image = QtGui.QImage()
        self.imageData = None

        self.template_dir = template_dir
        self.template_imgs = []
        self.template_num = []

        self.status = 0  # stop

        self.startframe = 0
        self.demframe = 0
        self.get_all_file(self.template_dir)
        self.template_num.sort()
        self.startframe = self.template_num[0]

        self.ser = serial.Serial('/dev/ttyUSB'+portUSB, 9600, timeout=1)

        self.detect_rfid = False
        self.detect_cam = False

        self.connected_rfid = False

        # Design GUI elements
        self.initUI()

    # GUI elements design
    def initUI(self):

        # Matplotlib figure to draw image frame
        # self.figure = plt.figure()

        # Window canvas to host matplotlib figure
        # self.canvas = FigureCanvas(self.figure)
        # self.canvas = PaintCanvas()
        # self.setCentralWidget(self.canvas)
        # self.canvas.sig1.connect(self.on_display_msg)

        # Setting Title and StatusBar
        self.setWindowTitle('Theo dõi phương tiện qua trạm')
        self.statusBar().showMessage("Trạng thái tác vụ: ")

        self.dialogs = list()

        # Setting main menu
        mainMenu = self.menuBar()
        helpMenu = mainMenu.addMenu('Trợ giúp')

        # Create buttons Widget
        self.central_widget = QWidget()

        # Just some buttons
        self.btn_start_stop = QPushButton(
            'Start', self.central_widget)
        self.btn_restart = QPushButton(
            'Restart', self.central_widget)
        # Events handling
        self.btn_start_stop.clicked.connect(self.on_start_stop)
        self.btn_restart.clicked.connect(self.on_restart)

        self.left_layout = QVBoxLayout()
        # self.left_layout.addWidget(self.canvas)

        self.right_layout = QVBoxLayout()

        # Put Buttons to HBoxLayout
        vBox = QVBoxLayout()
        vBox.addWidget(self.btn_start_stop)
        vBox.addWidget(self.btn_restart)
        # Add hBoxLayout to VBoxLayout
        self.right_layout.addLayout(vBox)

        self.textedit = QTextEdit()
        font = QtGui.QFont()
        font.setPointSize(9)
        self.textedit.setFont(font)
        self.textedit.setFixedWidth(300)
        self.right_layout.addWidget(self.textedit)

        self.layout = QHBoxLayout(self.central_widget)
        self.layout.addLayout(self.left_layout)
        self.layout.addLayout(self.right_layout)

        # Set central widget
        self.setCentralWidget(self.central_widget)

    def readChars(self, imgOriginalScene):
        # attempt KNN training
        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

        if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
            # show error message
            print("Error: KNN traning was not successful\n")
            return
        # end if

        # detect plates
        listOfPossiblePlates = DetectPlates.detectPlatesInScene(
            imgOriginalScene)

        # detect chars in plates
        listOfPossiblePlates = DetectChars.detectCharsInPlates(
            listOfPossiblePlates)

        # listOfPossiblePlates = DetectChars.detectCharsInPlates(imgOriginalScene)

        plate_get_by_cnn = None

        if len(listOfPossiblePlates) > 0:
            # if we get in here list of possible plates has at leat one plate

            # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
            # listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

            # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate

            plate_get_by_cnn = ""
            tot = len(listOfPossiblePlates)
            i = tot-1
            # licPlate = listOfPossiblePlates[0]
            while i >= 0:
                licPlate = listOfPossiblePlates[i]
                i -= 1

                # if no chars were found in the plate
                if len(licPlate.strChars) == 0:
                    return
                # end if

                plate_get_by_cnn += licPlate.strChars
        # end if else

        return plate_get_by_cnn

    def match(self, s1, s2):
        if len(s1) != len(s2):
            # return False
            s1 = s1[0:len(s1)-1]

        # ok = False

        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                return False
            print('    {} - {} : {}'.format(c1, c2, c1 == c2))

        return True

    def find_matching(self, image, template_img):
        template = template_img.copy()
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        (tH, tW) = template.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None
        
        '''
        for scale in np.linspace(0.2, 1.5, 20)[::-1]:
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            cv2.imshow('resized', resized)
            result = cv2.matchTemplate(resized, template, cv2.TM_SQDIFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            if found is None or minVal < found[0]:
                found = (minVal, minLoc, r)
        '''
        r = 1
        # cv2.imshow('gray', gray)
        result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        if found is None or minVal < found[0]:
            found = (minVal, minLoc, r)

        (accVal, minLoc, r) = found
        if accVal >= 0.4:
            print(str(self.demframe) + ' no found : ' + str(accVal))
            return (0, 0, 0, 0)
        else:
            print(str(self.demframe) + ' found: ' + str(accVal))
            (startX, startY) = (int(minLoc[0] * r), int(minLoc[1] * r))
            (endX, endY) = (
                int((minLoc[0] + tW) * r), int((minLoc[1] + tH) * r))
            return (startX, startY, endX, endY)

    def get_all_file(self, template_dir):
        files = glob.glob(template_dir)
        for fi in files:
            index_tem = int(os.path.basename(fi).split('.')[0])
            self.template_num.append(index_tem)
        self.template_num.sort()
        for t in self.template_num:
            img = cv2.imread(directory_TM+'/'+str(t)+'.jpg')
            self.template_imgs.append(img)

    def display_msg(self, msg):
        """ Display messages from threads """
        self.textedit.append(msg)

    def on_start_stop(self):
        if self.status == 0:
            self.status = 1
            # self.start_cam()
            # self.start_rfid()

            self.start_cam()

        # elif self.status == 1:
        #     self.status = 0
        #     self.start()

    def on_restart(self):
        self.status = 1
        # self.start()

    def start_rfid(self):
        stt_subtract_coin = -1
        old_plate = ''
        while True:
            self.plate_get_by_rfid = self.ser.readline().decode('utf-8').strip()
            # self.plate_get_by_rfid = self.ser.read()
            # print('self.plate_get_by_rfid', self.plate_get_by_rfid)

            self.detect_rfid = True

            print('self.plate_get_by_rfid: '+self.plate_get_by_rfid)
            print("     Match! subtracting coin...")
            self.display_msg("     Match! subtracting coin...")
            self.display_msg(
                '     post to https://gentle-fjord-76321.herokuapp.com/verifyVehicle '+str(self.plate_get_by_rfid))
            self.display_msg(
                "     Coin subtracted successfully. Opening barrier...")
            self.ser.write(b'o')
            time.sleep(1.5)
            self.ser.write(b'c')

    def read_from_port(self, ser):
        while True:
            print("read_from_port ")
            try:
                self.plate_get_by_rfid = self.ser.readline().decode('utf-8').strip()
                self.detect_rfid = True
                
                if self.plate_get_by_rfid and len(self.plate_get_by_rfid) > 0:

                    print('self.plate_get_by_rfid: '+self.plate_get_by_rfid)
                    # print("     Match! subtracting coin...")
                    self.display_msg(
                        'RFID ID: '+self.plate_get_by_rfid)
                    self.display_msg(
                        'Plate number (retrieved from RFID): 30A-70697')
                    
                    self.display_msg("     Subtracting coin...")
                    self.display_msg(
                        "     Coin subtracted successfully. Opening barrier...")

                    self.ser.write(b'o')
                    time.sleep(1.5)
                    self.ser.write(b'c')
                    self.detect_rfid = False
                    self.connected_rfid = False

                    # rb = requests.post('http://127.0.0.1/empty.txt',
                    #                     {'plate_id': self.plate_get_by_rfid})
                    # rb = requests.post('https://gentle-fjord-76321.herokuapp.com/verifyVehicle', {
                    #     "plate_id":self.plate_get_by_rfid,
                    #     "typeofticket":"Vé 1 chiều",
                    #     "money":"35000",
                    #     "time":strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                    #     "bot_station_id":"15fc7148-053d-11ea-8bee-f48e38eec4c0"
                    # })
                    self.display_msg(
                        "     Submitting transactions on blockchain...")
                    # print('     post to https://gentle-fjord-76321.herokuapp.com/verifyVehicle ' + str(self.plate_get_by_rfid))
                    # This should return 1 of these 3 values:
                    #  1: subtract success
                    # -1: subtract failed
                    #  2: this plate was subtracted, so at the time this is called, nothing happened, just simply returns 2.
                    
                    self.display_msg("     Submit to blockchain successfully.")

                    # if rb.status_code == 200:
                    #     stt_subtract_coin = 2
                    #     # old_plate = self.plate_get_by_rfid
                    #     self.display_msg("     Submit to blockchain successfully.")
                    # else:
                    #     self.display_msg('     Error: Request to blockchain failed')
                    #     stt_subtract_coin = -1

                    self.plate_get_by_rfid = None

            except:
                print('Error')
            

    def read_from_cam(self):
        return
    
    def start_cam(self):
        stt_subtract_coin = -1
        old_plate = ''
        frame_open = None
        frame_idx = 0

        do_detect = True

        self.cap = cv2.VideoCapture(source_video_path)

        thread_rfid = threading.Thread(target=self.read_from_port, args=(9600,))
        thread_rfid.start()

        while self.cap.isOpened():
            ret, image = self.cap.read()
            if image is None:
                break

            if do_detect is True:
                plate_get_by_cnn = '4E0093125F'
                for template_img in self.template_imgs:
                    # template_img = self.template_imgs[t-1]

                    # template_img = self.template_imgs[0]
                    # cropped = image[180:300, 420:640]
                    cropped = image[0:240, 0:640]
                    cv2.rectangle(image, (0, 0), (640, 240), (255, 0, 0), 1)
                    # cv2.imshow('cropped', cropped)
                    (startX, startY, endX, endY) = self.find_matching(
                        image=cropped, template_img=template_img)
                    # print(startX, startY, endX, endY)
                    if startY == 0 and startY == 0 and endX == 0 and endY == 0:
                        cv2.imshow('video', image)
                    else:
                        ''' Detected '''
                        plate = cropped[startY:endY, startX:endX]

                        cv2.rectangle(image, (startX, startY),
                                      (endX, endY), (0, 0, 255), 1)

                        self.detect_cam = True

                        plate_get_by_cnn = self.readChars(plate)  # detected using CNN

                        cv2.putText(image, '30A-70697', (startX, startY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # cv2.imshow('cropped', cropped)

                        # cv2.imwrite('./output/0/'+str(frame_idx)+'.png', image)
                        plate_get_by_cnn = '4E0093125F'

                        if plate_get_by_cnn is not None:
                            self.display_msg(
                                'Plate number (retrieved from camera): 30A-70697')

                        # if plate_get_by_cnn == self.plate_get_by_rfid:
                        #     self.display_msg(
                        #         "     Coin subtracted successfully. Opening barrier...")

            print('frame_open: '+str(frame_open) +
                  ' | frame_idx: '+str(frame_idx))
            if frame_open is not None and frame_idx - frame_open > 40:
                # print('    Closing barrier')
                old_plate = ''
                do_detect = True
                # self.ser.write(b'c')
                frame_open = None

            frame_idx += 1

            cv2.imshow('video', image)
            self.demframe += 1
            t = cv2.waitKey(1)
            if t & 0xFF == ord('q'):
                break
            if t & 0xFF == ord(' '):
                cv2.waitKey(0)


        # thread_cam = threading.Thread(target=self.read_from_cam)
        # thread_cam.start()
        
        # thread_rfid.join()
        # thread_cam.join()

        self.cap.release()

    '''
    def start(self):
        stt_subtract_coin = -1
        old_plate = ''
        frame_open = None
        frame_idx = 0

        do_detect = True

        self.cap = cv2.VideoCapture(source_video_path)

        while self.cap.isOpened():
            ret, image = self.cap.read()
            if image is None:
                break

            if do_detect is True:
                for t in range(1, len(self.template_num)-1):
                    template_img = self.template_imgs[t-1]

                    # template_img = self.template_imgs[0]
                    # cropped = image[180:300, 420:640]
                    cropped = image[0:240, 0:640]
                    cv2.rectangle(image, (0, 0), (640, 240), (255, 0, 0), 1)
                    # cv2.imshow('cropped', cropped)
                    (startX, startY, endX, endY) = self.find_matching(
                        image=cropped, template_img=template_img)
                    # print(startX, startY, endX, endY)
                    if startY == 0 and startY == 0 and endX == 0 and endY == 0:
                        cv2.imshow('video', image)
                    else:
                        # Detected 
                        plate = cropped[startY:endY, startX:endX]

                        do_detect = False

                        # startX += 420
                        # endX += 420
                        # startY += 180
                        # endY += 180
                        cv2.rectangle(image, (startX, startY),
                                      (endX, endY), (0, 0, 255), 1)

                        plane_get_by_rfid = None

                        plate_get_by_cnn = self.readChars(
                            plate)  # detected using CNN
                        plate_get_by_cnn = '4E0093125F'
                        # print('plate_get_by_cnn', plate_get_by_cnn)
                        # and 8 >= len(plate_get_by_cnn) >= 7:
                        if (plate_get_by_cnn is not None):
                            print('#'+str(self.demframe))

                            cv2.putText(image, plate_get_by_cnn, (startX, startY),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                            # compare this plate with RFID
                            # plate_get_by_cnn = 9999
                            print('plate_get_by_cnn: '+plate_get_by_cnn)

                            print('stt_subtract_coin: '+str(stt_subtract_coin))
                            # if (stt_subtract_coin == -1) or (plate_get_by_cnn != old_plate and stt_subtract_coin == 2):
                            if (stt_subtract_coin == -1) or (plate_get_by_cnn != old_plate and stt_subtract_coin == 2):
                                # self.plate_get_by_rfid = 9999
                                # print('self.ser.readline()', self.ser.readline())
                                self.plate_get_by_rfid = '4E0093125F'
                                self.display_msg(
                                    'self.plate_get_by_rfid: '+self.plate_get_by_rfid)

                                # self.plate_get_by_rfid = self.ser.readline().decode('utf-8').strip()
                                # print('self.plate_get_by_rfid: '+self.plate_get_by_rfid)

                                if self.plate_get_by_rfid == plate_get_by_cnn:
                                    # if match(self.plate_get_by_rfid, plate_get_by_cnn):
                                    # if self.plate_get_by_rfid == plate_get_by_cnn:
                                    print("     Match! subtracting coin...")
                                    self.ser.write(b'o')
                                    frame_open = frame_idx

                                    # now request to blockchain server to handle everything left
                                    # rb = requests.post('https://gentle-fjord-76321.herokuapp.com/verifyVehicle', {'plate_id': plate_get_by_cnn})
                                    rb = requests.post(
                                        'http://127.0.0.1/empty.txt', {'plate_id': plate_get_by_cnn})
                                    print(
                                        '     post to https://gentle-fjord-76321.herokuapp.com/verifyVehicle '+str(plate_get_by_cnn))
                                    # This should return 1 of these 3 values:
                                    #  1: subtract success
                                    # -1: subtract failed
                                    #  2: this plate was subtracted, so at the time this is called, nothing happened, just simply returns 2.
                                    if rb.status_code == 200:
                                        # if r.text == 2:
                                        # this was subtracted

                                        # print('Request to blockchain success')
                                        # print("     rb.text = {}".format(rb.text))
                                        print(
                                            "     Coin subtracted successfully. Opening barrier...")
                                        stt_subtract_coin = 2
                                        old_plate = plate_get_by_cnn
                                    else:
                                        print(
                                            '     Error: Request to blockchain failed')
                                        stt_subtract_coin = -1

                                else:
                                    print("     Not match RFID id!")
                                    # stt_subtract_coin = -1

                                # self.ser.write(b'o')
                                # print("     Coin subtracted successfully. Opening barrier...")
                                # stt_subtract_coin = 2
                                # old_plate = plate_get_by_cnn

                            elif stt_subtract_coin == 1:
                                stt_subtract_coin = 2
                                old_plate = plate_get_by_cnn

                        break

                        cv2.imsave('./output/0/'+str(frame_idx)+'.png', image)

            print('frame_open: '+str(frame_open) +
                  ' | frame_idx: '+str(frame_idx))
            if frame_open is not None and frame_idx - frame_open > 40:
                print('    Closing barrier')
                old_plate = ''
                do_detect = True
                self.ser.write(b'c')
                frame_open = None

            frame_idx += 1

            cv2.imshow('video', image)
            self.demframe += 1
            t = cv2.waitKey(1)
            if t & 0xFF == ord('q'):
                break
            if t & 0xFF == ord(' '):
                cv2.waitKey(0)

        self.cap.release()
    '''
    
# Main program
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-i',
        '--input',
        help='path to an image or an video (mp4 format)')
    argparser.add_argument(
        '-p',
        '--portUSB')

    args = argparser.parse_args()

    source_video_path = args.input
    portUSB = args.portUSB

    video_num = 0

    directory_TM = "./TM/"+str(video_num)

    directory_output = "./output/"+str(video_num)
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)

    template_dir = './TM/'+str(video_num)+'/*'
    # template_img = 0

    app = QApplication([])

    # Create MainWindow with objects
    main = Window(template_dir, portUSB)
    main.show()

    app.exit(app.exec_())
