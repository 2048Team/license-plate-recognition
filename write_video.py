#!/usr/bin/python

import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import os
from pytesseract import image_to_string
import scipy.fftpack  # For FFT2
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    frame_width = int(cap.get(3))  # float
    frame_height = int(cap.get(4))  # float

    out = cv2.VideoWriter('output/'+source_video_path.split('/')
                        [-1], fourcc, 80.0, (frame_width, frame_height), True)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if demframe in listframe:
            index = listframe.index(demframe)
            x0, y0, x1, y1 = listrect[index]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1)
            
            plate = frame[y0:y1, x0:x1]


        cv2.imshow('video', frame)
        out.write(frame)
        demframe += 1
        t = cv2.waitKey(1)
        if t & 0xFF == ord('q'):
            break
    f.close()
    out.release()
    cap.release()


def process_cropped_plate(imgpath):
    # Read in image
    #img = cv2.imread('5DnwY.jpg', 0)
    img = cv2.imread(imgpath, 0)

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Remove some columns from the beginning and end
    #img = img[:, 59:cols-20]

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1*Ioutlow[0:rows, 0:cols] + gamma2*Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")

    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < 70
    Ithresh = 255*Ithresh.astype("uint8")

    # Clear off the border.  Choose a border radius of 5 pixels
    Iclear = imclearborder(Ithresh, 5)

    # Eliminate regions that have areas below 120 pixels
    Iopen = bwareaopen(Iclear, 6)

    print(image_to_string(Iopen, lang='eng'))

    # Show all images
    cv2.imshow('Original Image', img)
    cv2.imshow('Homomorphic Filtered Result', Ihmf2)
    cv2.imshow('Thresholded Result', Ithresh)
    cv2.imshow('Opened Result', Iopen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# imclearborder definition
def imclearborder(imgBW, radius=2):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    # _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
    #                                           cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]

    contourList = []  # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (
                rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (
                colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

    return imgBWcopy

# bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    # _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
    #                                           cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

    return imgBWcopy



if __name__ == '__main__':
    args = argparser.parse_args()
    __main__(args)
