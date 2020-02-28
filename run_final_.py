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


# Change this to True for debugging
showSteps = False

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def requestToBlockchain(plate_chars):
    r = requests.get('https://gentle-fjord-76321.herokuapp.com/verifyVehicle')
    #r = requests.get('http://192.168.43.42:8080')
    # This should return 1 of these 3 values:
    #  1: subtract success
    # -1: subtract failed
    #  2: this plate was subtracted, so at the time this is called, nothing happened, just simply returns 2.
    if r.status_code == 200:
        # if r.text == 2:
        # this was subtracted

        # print('Request to blockchain success')
        print("     Coin subtracted successfully")
    else:
        print('     Error: Request to blockchain failed')

    return r.text


def compareAndSend(plate_chars, is_last=False, status_prev=False):
    # send request to compare plate_chars detected in image with RFID card
    #r = requests.get('http://192.168.43.182/rfid/input.txt')
    r = requests.get('http://192.168.8.102/rfid/input.txt')
    """
    if r.encoding is None or r.encoding == 'ISO-8859-1':
        r.encoding = r.apparent_encoding
    """

    if r.status_code == 200:
        # if match
        print(r.text)
        if r.text == plate_chars:
            print("     Match! subtracting coin...")

            # now request to blockchain server to handle everything left
            return requestToBlockchain(plate_chars)
        else:
            print("     Not match RFID id!")
            return -1
    else:
        print('     Error: Request to RFID server failed!')

        return -1


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

    plate_chars = None

    if len(listOfPossiblePlates) > 0:
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        # listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate

        plate_chars = ""
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

            plate_chars += licPlate.strChars
    # end if else

    return plate_chars



def match(s1, s2):
    if len(s1) != len(s2):
        #return False
        s1 = s1[0:len(s1)-1]

    #ok = False

    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            return False
        print('    {} - {} : {}'.format(c1, c2, c1 == c2))
    
    return True


def __main__(args):
    source_video_path = args.input
    video_num = source_video_path.split('.')[0].split('/')[-1]

    f = open('output/'+str(video_num)+'/frames.txt', 'r')
    listframe = []
    listrect = []
    for line in f:
        (idf, ids, x0, y0, x1, y1) = line.split(' ')
        y1 = y1.split('\n')[0]
        listframe.append(int(idf))
        listrect.append((int(x0), int(y0), int(x1), int(y1)))
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
        if demframe in listframe:
            index = listframe.index(demframe)
            x0, y0, x1, y1 = listrect[index]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)

            plate = frame[y0:y1, x0:x1]

            plate_chars = readChars(plate)
            if (plate_chars is not None) and 8 >= len(plate_chars) >= 7:
                print(str(demframe)+": "+plate_chars)

                cv2.putText(frame, plate_chars, (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # compare this plate with RFID
                is_last = True if (index == len(listframe)-1) else False

                """
                if (stt_subtract_coin == -1) or (plate_chars != old_plate and stt_subtract_coin == 2):
                    stt_subtract_coin = compareAndSend(
                        plate_chars, is_last, stt_subtract_coin)

                    if stt_subtract_coin != -1:
                        old_plate = plate_chars

                    print(stt_subtract_coin)
                elif stt_subtract_coin == 1:
                    stt_subtract_coin = 2
                    old_plate = plate_chars
                """

                if (stt_subtract_coin == -1) or (plate_chars != old_plate and stt_subtract_coin == 2):
                    r = requests.get('http://192.168.8.102/rfid/input.txt')
                    if r.status_code == 200:
                        plate_get = r.text.split("\n")
                        plate_get = plate_get[0]
                        #match(plate_get, plate_chars)
                        # if match
                        plate_get = plate_get[0:len(plate_get)-1]
                        if plate_get == plate_chars:
                        #if match(plate_get, plate_chars):
                        #if plate_get == plate_chars:
                            print("     Match! subtracting coin...")

                            # now request to blockchain server to handle everything left
                            rb = requests.get('http://192.168.8.100:8080/?plate_id='+plate_chars)
                            print('http://192.168.8.100:8080/?plate_id='+plate_chars)
                            # This should return 1 of these 3 values:
                            #  1: subtract success
                            # -1: subtract failed
                            #  2: this plate was subtracted, so at the time this is called, nothing happened, just simply returns 2.
                            if rb.status_code == 200:
                                # if r.text == 2:
                                # this was subtracted

                                # print('Request to blockchain success')
                                #print("     rb.text = {}".format(rb.text))
                                print("     Coin subtracted successfully")
                                stt_subtract_coin = 2
                                old_plate = plate_chars
                            else:
                                print('     Error: Request to blockchain failed')
                                stt_subtract_coin = -1

                        else:
                            print("     Not match RFID id!")
                            #stt_subtract_coin = -1
                    else:
                        print('     Error: Request to RFID server failed!')
                        #stt_subtract_coin = -1
                elif stt_subtract_coin == 1:
                    stt_subtract_coin = 2
                    old_plate = plate_chars


                file = open("output/plate_chars.txt", "w")
                file.write(plate_chars)
                file.close()
            cv2.waitKey(8)
        else:
            cv2.waitKey(60)

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
