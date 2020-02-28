#!/usr/bin/python

# import numpy as np
import os
import cv2
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def __main__(args):
    source_video_path = args.input

    video_num = 0
    # video_num = source_video_path.split('.')[0].split('/')[-1]
    directory_TM = "./TM/"+str(video_num)
    if not os.path.exists(directory_TM):
        os.makedirs(directory_TM)

    cap = cv2.VideoCapture(source_video_path)
    demframe = 0
    play = True
    #count_tem = 1
    print('directory_TM', directory_TM)
    while(cap.isOpened()):
        if play:
            ret, frame = cap.read()
            print(demframe)
            demframe += 1
            
            if frame is None:
                break
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == 32:
                # play = not play
                play = False
        else:
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite(directory_TM+'/'+str(demframe)+'.jpg',frame)
            elif k == 32:
                play = True
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = argparser.parse_args()
    __main__(args)
