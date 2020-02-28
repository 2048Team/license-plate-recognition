#!/usr/bin/python

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

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

args = argparser.parse_args()

source_video_path = args.input
idbien = 3

video_num = 0

directory_TM = "./TM/"+str(video_num)

directory_output = "./output/"+str(video_num)
if not os.path.exists(directory_output):
    os.makedirs(directory_output)


template_imgs = []
template_num = []
template_dir = './TM/'+str(video_num)+'/*'
# template_img = 0

# 3.avi
if video_num == 3:
    endframe = 220
# 4.avi
elif video_num == 4:
    endframe = 160
# 5.avi
elif video_num == 5:
    endframe = 226

cap = cv2.VideoCapture(source_video_path)
startframe = 0
#endframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
demframe = 0


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




def get_all_file(template_dir):
    files = glob.glob(template_dir)
    for fi in files:
        index_tem = int(os.path.basename(fi).split('.')[0])
        template_num.append(index_tem)
    template_num.sort()
    for t in template_num:
        img = cv2.imread(directory_TM+'/'+str(t)+'.jpg')
        template_imgs.append(img)


def find_matching(image, template_img):
    global demframe
    template = template_img.copy()
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    (tH, tW) = template.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    for scale in np.linspace(0.2, 1.5, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        result = cv2.matchTemplate(resized, template, cv2.TM_SQDIFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        if found is None or minVal < found[0]:
            found = (minVal, minLoc, r)
    (accVal, minLoc, r) = found
    if accVal >= 0.09:
        print(str(demframe) + ' no found : ' + str(accVal))
        return (0, 0, 0, 0)
    else:
        print(str(demframe) + ' : ' + str(accVal))
        (startX, startY) = (int(minLoc[0] * r), int(minLoc[1] * r))
        (endX, endY) = (int((minLoc[0] + tW) * r), int((minLoc[1] + tH) * r))
        return (startX, startY, endX, endY)


get_all_file(template_dir)
template_num.sort()
startframe = template_num[0]

ser = serial.Serial('/dev/ttyUSB0',9600)

stt_subtract_coin = -1
old_plate = ''
frame_open = None
frame_idx = 0

while cap.isOpened():
    ret, image = cap.read()
    if image is None:
        break

    frame_idx += 1
    
    '''
    # chon vung quan tam
    if endframe >= demframe >= startframe:
        for t in range(1, len(template_num)-1):
            if template_num[t] > demframe >= template_num[t-1]:
                template_img = template_imgs[t-1]
        if demframe >= template_num[-1]:
            template_img = template_imgs[-1]
        (startX, startY, endX, endY) = find_matching(
            image=image, template_img=template_img)
        if startY == 0 and startY == 0 and endX == 0 and endY == 0:
            cv2.imshow('video', image)
        else:
            imageroi = image[startY:endY, startX:endX]
            # cv2.imwrite(directory_extract+"/t" + str(demframe)+".jpg", imageroi)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)
            # f.write('{} {} {} {} {} {}\n'.format(demframe, idbien, startX, startY, endX, endY))
    '''

    # for t in range(1, len(template_num)-1):
    #     template_img = template_imgs[t-1]
    template_img = template_imgs[0]
    cropped = image[180:300, 420:640]
    (startX, startY, endX, endY) = find_matching(image=cropped, template_img=template_img)
    if startY == 0 and startY == 0 and endX == 0 and endY == 0:
        cv2.imshow('video', image)
    else:
        ''' Detected '''
        plate = cropped[startY:endY, startX:endX]
        
        startX += 420
        endX += 420
        startY += 180
        endY += 180
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)
        
        plane_get_by_rfid = None
        if True:
            plate_get_by_cnn = readChars(plate)  # detected using CNN
            plate_get_by_cnn = '4E0093125F'
            # print('plate_get_by_cnn', plate_get_by_cnn)
            if (plate_get_by_cnn is not None): #and 8 >= len(plate_get_by_cnn) >= 7:
                print('#'+str(demframe))

                cv2.putText(image, plate_get_by_cnn, (startX, startY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # compare this plate with RFID
                # plate_get_by_cnn = 9999
                print('plate_get_by_cnn', plate_get_by_cnn)

                print('stt_subtract_coin', stt_subtract_coin)
                # if (stt_subtract_coin == -1) or (plate_get_by_cnn != old_plate and stt_subtract_coin == 2):
                if (stt_subtract_coin == -1) or (plate_get_by_cnn != old_plate and stt_subtract_coin == 2):
                    # plate_get_by_rfid = 9999
                    # print('ser.readline()', ser.readline())
                    plate_get_by_rfid = ser.readline().decode('utf-8').strip()
                    print('plate_get_by_rfid', plate_get_by_rfid)
                    
                    if plate_get_by_rfid == plate_get_by_cnn:
                        # if match(plate_get_by_rfid, plate_get_by_cnn):
                        # if plate_get_by_rfid == plate_get_by_cnn:
                        print("     Match! subtracting coin... Opening barrier")
                        ser.write(b'o')
                        frame_open = frame_idx

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
        
    print('frame_open', frame_open)
    print('frame_idx', frame_idx)
    print('frame_open', frame_open)
    if frame_open is not None and frame_idx - frame_open > 5:
        print('    Closing barrier')
        old_plate = ''
        ser.write(b'c')
        frame_open = None
        
    frame_idx += 1
        

    cv2.imshow('video', image)
    demframe += 1
    t = cv2.waitKey(1)
    if t & 0xFF == ord('q'):
        break
    if t & 0xFF == ord(' '):
        cv2.waitKey(0)

cap.release()

