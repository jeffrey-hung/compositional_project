#!/usr/bin/python3

import numpy as np
import cv2
import dlib
from math import hypot
import time
import sys
import os
import vlc
## ================ resources ================

## https://medium.com/p/89c79f0a246a/responses/show
## https://medium.com/@nuwanprabhath/installing-opencv-in-macos-high-sierra-for-python-3-89c79f0a246a
## https://www.learnopencv.com/install-dlib-on-macos/

## ===========================================

def get_gaze_ratio(frame, landmarks, lowest_point):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	eye_region = np.array([(landmarks.part(lowest_point).x, landmarks.part(lowest_point).y),
							(landmarks.part(lowest_point+1).x, landmarks.part(lowest_point+1).y),
							(landmarks.part(lowest_point+2).x, landmarks.part(lowest_point+2).y),
							(landmarks.part(lowest_point+3).x, landmarks.part(lowest_point+3).y),
							(landmarks.part(lowest_point+4).x, landmarks.part(lowest_point+4).y),
							(landmarks.part(lowest_point+5).x, landmarks.part(lowest_point+5).y)], np.int32)

	# Mask out only the left eye
	height, width, _ = frame.shape
	mask = np.zeros((height,width),np.uint8)
	cv2.polylines(mask, [eye_region], True, 255, 2)
	cv2.fillPoly(mask, [eye_region], 255)
	eye = cv2.bitwise_and(gray,gray, mask=mask)

	# segregate left eye from face
	min_x = np.min(eye_region[:,0])
	max_x = np.max(eye_region[:,0])
	min_y = np.min(eye_region[:,1])
	max_y = np.max(eye_region[:,1])

	gray_eye = eye[min_y:max_y, min_x:max_x]
	_, threshold_eye=cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

	height, width = threshold_eye.shape

	# get amount of white (right side, left side) in left eye 
	left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
	left_side_white = cv2.countNonZero(left_side_threshold)

	right_side_threshold = threshold_eye[0:height, int(width/2):width]	
	right_side_white = cv2.countNonZero(right_side_threshold)

	gaze_ratio = left_side_white+right_side_white
	if left_side_white == 0:
		gaze_ratio = 1
	elif right_side_white == 0:
		gaze_ratio = 5
	else:
		gaze_ratio = left_side_white / right_side_white
	return gaze_ratio

def eye_track():
	cap = cv2.VideoCapture(0) #capture video from webcam
	detector = dlib.get_frontal_face_detector() #find face
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # find face details
	font = cv2.FONT_HERSHEY_SIMPLEX #debugging and testing features
	start = time.time()
	trigger = False
	
	
	while True:

		_, frame = cap.read() #grab image from image capture
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray) # array of all the faces
			
		for face in faces:

			landmarks = predictor(gray,face)

			################################## Gaze detection ######################################

			left_gaze_ratio = get_gaze_ratio(frame, landmarks, 36)
			right_gaze_ratio = get_gaze_ratio(frame, landmarks, 42)
			gaze_ratio = (right_gaze_ratio + left_gaze_ratio) / 2

			################################## Timer ######################################
			print(gaze_ratio)
			if gaze_ratio > 0.85 and gaze_ratio < 1.4:
				start = time.time()
			end = time.time()
			
			if end-start>2:
				print("################################## PAY ATTENTION ######################################")
				trigger = True
				break
				start = time.time()

		#cv2.imshow("Frame", frame) #display  image
		key = cv2.waitKey(1)
		if key == 27 or trigger == True:
			break
	# finishing up
	cap.release()
	cv2.destroyAllWindows()

def start_video(path):
	Instance = vlc.Instance('--fullscreen')
	player = Instance.media_player_new()
	video = Instance.media_new(path)
	video.get_mrl()
	player.set_media(video)
	return player

if __name__ == "__main__":

	video_A = "test.mp4"
	video_B = "test.mp4"
	
	player_A = start_video(video_A)
	player_A.play()
	eye_track()
	player_A.stop()

	player_B = start_video(video_B)
	player_B.play()
	eye_track()
	player_B.stop()