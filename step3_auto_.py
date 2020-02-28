#!/usr/bin/python

import numpy as np
import argparse
import imutils
import os
import glob
import cv2
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

args = argparser.parse_args()

source_video_path = args.input
idbien = 3

video_num = int(source_video_path.split('.')[0].split('/')[-1])

directory_TM = "./TM/"+str(video_num)

directory_output = "./output/"+str(video_num)
if not os.path.exists(directory_output):
    os.makedirs(directory_output)

directory_extract = "./extract/"+str(video_num)
if not os.path.exists(directory_extract):
    os.makedirs(directory_extract)


template_imgs = []
template_num = []
template_dir = './TM/'+str(video_num)+'/*'
template_img = 0

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
f = open(directory_output+'/frames.txt', 'w')


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
    if accVal >= 0.18:
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

while cap.isOpened():
    ret, image = cap.read()
    if image is None:
        break

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

    cv2.imshow('video', image)
    demframe += 1
    t = cv2.waitKey(1)
    if t & 0xFF == ord('q'):
        break
    if t & 0xFF == ord(' '):
        cv2.waitKey(0)

cap.release()
f.close()
