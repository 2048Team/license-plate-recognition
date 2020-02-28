#!/usr/bin/python

import cv2
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def __main__(args):
    #file_path      =       'data/6.mp4'
    file_path       =       args.input

    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(3))  # float
    frame_height = int(cap.get(4)) # float
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_path.split('.')[0]+'.avi', fourcc, 80.0, (frame_width, frame_height),True)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break

        #frame = cv2.resize(frame,(640,480))

        # Rotate the frame
        (h, w) = frame.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2)
        angle = -90
        scale = 1.0
        # do the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        frame = cv2.warpAffine(frame, M, (w, h)) 

        #frame = imutils.resize(frame, width=450)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = np.dstack([frame, frame, frame])

        cv2.imshow('video',frame)
        out.write(frame)

        t = cv2.waitKey(1)
        if t & 0xFF == ord('q'):
            break
            
    out.release()
    cap.release()


if __name__ == '__main__':
    args = argparser.parse_args()
    __main__(args)
