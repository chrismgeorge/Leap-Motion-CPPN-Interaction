from PIL import Image
import numpy as np 
import cv2
import time
import os
import random
import pdb

from leap import *


def extract_frames(vid_file):
	cap = cv2.VideoCapture(vid_file)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frames = [None for _ in range(frame_count)]
	i = 0
	print("reading " + str(frame_count) + " frames from " + vid_file)
	for i in range(frame_count):
	    ret, frame = cap.read()

	    frames[i] = frame
	    i += 1
	cap.release()
	return frames

def load_videos(num):
	folder = "./videos/" + str(num)
	img1 = folder + "/1.mp4"
	img2 = folder + "/2.mp4"
	return extract_frames(img1), extract_frames(img2)

def open_window():
	print("opening window")
	fullscreen = False
	if fullscreen:
		cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	else:
		cv2.namedWindow("window")

def close_window():
	print("closing window")
	cv2.destroyAllWindows()

def blend(img1, img2):
	x = min(img1.shape[0], img2.shape[0])
	y = min(img1.shape[1], img2.shape[1])
	A = img1[:x, :y, :]
	B = img2[:x, :y, :]
	# out = (A / 2 + B / 2)
	out = np.maximum(A, B)
	return out




def main():

	vid1, vid2 = load_videos(0)

	open_window()

	i1 = 0
	i2 = 0
	len1 = len(vid1)
	len2 = len(vid2)

	while(True):

		if not LEAP:
			i1,i2 = (i1+1)%len1,(i2+1)%len2
		else:
			data = getVideoIndicesFromLeap(leapController)
			if data is None:
				continue
			else:
				i1,i2 = data
				i1 = i1 % len1
				i2 = i2 % len2

		cv2.imshow("window", blend(vid1[i1], vid2[i2]))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	close_window()



if __name__ == '__main__':
	main()

