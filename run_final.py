#!/usr/bin/python

import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import os
from pytesseract import image_to_string
import scipy.fftpack  # For FFT2

import DetectChars
import DetectPlates
import PossiblePlate

import requests
import argparse

import serial


# Change this to True for debugging
showSteps = False

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')



def readChars(imgOriginalScene):
    # attempt KNN training
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("Error: KNN traning was not successful\n")  # show error message
        return
    # end if

    # detect plates
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

    # detect chars in plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(
        listOfPossiblePlates)

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


def match(s1, s2):
    if len(s1) != len(s2):
        # return False
        s1 = s1[0:len(s1)-1]

    # ok = False

    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            return False
        print('    {} - {} : {}'.format(c1, c2, c1 == c2))

    return True


def __main__(args):
    source_video_path = args.input
    video_num = source_video_path.split('.')[0].split('/')[-1]

    # f = open('output/'+str(video_num)+'/frames.txt', 'r')
    # listframe = []
    # listrect = []
    # for line in f:
    #     (idf, ids, x0, y0, x1, y1) = line.split(' ')
    #     y1 = y1.split('\n')[0]
    #     listframe.append(int(idf))
    #     listrect.append((int(x0), int(y0), int(x1), int(y1)))
    demframe = 0

    cap = cv2.VideoCapture(source_video_path)
    cap.set(cv2.CAP_PROP_FPS, 10)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    frame_width = int(cap.get(3))  # float
    frame_height = int(cap.get(4))  # float

    out = cv2.VideoWriter('output/'+source_video_path.split('/')
                          [-1], fourcc, 80.0, (frame_width, frame_height), True)

    stt_subtract_coin = -1
    old_plate = ''

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        '''
        if demframe in listframe:
            index = listframe.index(demframe)
            x0, y0, x1, y1 = listrect[index]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)

            plate = frame[y0:y1, x0:x1]

            plate_get_by_cnn = readChars(plate)  # detected using CNN
            if (plate_get_by_cnn is not None) and 8 >= len(plate_get_by_cnn) >= 7:
                print(str(demframe)+": "+plate_get_by_cnn)

                cv2.putText(frame, plate_get_by_cnn, (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # compare this plate with RFID
                is_last = True if (index == len(listframe)-1) else False

                plate_get_by_cnn = 9999

                if (stt_subtract_coin == -1) or (plate_get_by_cnn != old_plate and stt_subtract_coin == 2):
                    plate_get_by_rfid = 9999
                    if plate_get_by_rfid == plate_get_by_cnn:
                        # if match(plate_get_by_rfid, plate_get_by_cnn):
                        # if plate_get_by_rfid == plate_get_by_cnn:
                        print("     Match! subtracting coin...")

                        # now request to blockchain server to handle everything left
                        rb = requests.post('https://gentle-fjord-76321.herokuapp.com/verifyVehicle', {'plate_id': plate_get_by_cnn})
                        print('     post to https://gentle-fjord-76321.herokuapp.com/verifyVehicle '+str(plate_get_by_cnn))
                        # This should return 1 of these 3 values:
                        #  1: subtract success
                        # -1: subtract failed
                        #  2: this plate was subtracted, so at the time this is called, nothing happened, just simply returns 2.
                        if rb.status_code == 200:
                            # if r.text == 2:
                            # this was subtracted

                            # print('Request to blockchain success')
                            # print("     rb.text = {}".format(rb.text))
                            print("     Coin subtracted successfully")
                            stt_subtract_coin = 2
                            old_plate = plate_get_by_cnn
                        else:
                            print('     Error: Request to blockchain failed')
                            stt_subtract_coin = -1

                    else:
                        print("     Not match RFID id!")
                        # stt_subtract_coin = -1
                elif stt_subtract_coin == 1:
                    stt_subtract_coin = 2
                    old_plate = plate_get_by_cnn

                # file = open("output/plate_get_by_cnn.txt", "w")
                # file.write(str(plate_get_by_cnn))
                # file.close()
            
            cv2.waitKey(1)
        else:
            cv2.waitKey(1)
        '''
        cv2.waitKey(1)

        cv2.imshow('video', frame)
        out.write(frame)
        demframe += 1
        t = cv2.waitKey(1)
        if t & 0xFF == ord('q'):
            break
    f.close()
    out.release()
    cap.release()


if __name__ == "__main__":
    args = argparser.parse_args()
    __main__(args)
