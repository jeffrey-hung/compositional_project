#!/usr/bin/python3

import numpy as np
import cv2
import dlib
from math import hypot

## ================ resources ================

## https://medium.com/p/89c79f0a246a/responses/show
## https://medium.com/@nuwanprabhath/installing-opencv-in-macos-high-sierra-for-python-3-89c79f0a246a
## https://www.learnopencv.com/install-dlib-on-macos/

## ===========================================

cap = cv2.VideoCapture(0) #capture video from webcam

detector = dlib.get_frontal_face_detector() #find face
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # find face details
font = cv2.FONT_HERSHEY_SIMPLEX #debugging and testing features

# calculate midpoint for eye landmarks
def midpoint(p1,p2):
	#return values must be ints as there cant be half pixels
	return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# calculate ratio for blinking between horizontal line/vertical (vertical decreases, hor stays the same, ratio changes)
def get_blink_ratio(eye_points, facial_landmarks):
	## get points eye (left/right dependant on eye_points)
	left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
	right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
	center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
	center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

	## draw cross section of right eye
	horizontal_line= cv2.line(frame, left_point, right_point, (0,255,0),2)
	vertical_line= cv2.line(frame, center_top, center_bottom, (0,255,0),2)
	
	# get length of two lines
	ver_line_length = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))
	hor_line_length = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1]))

	ratio = hor_line_length/ver_line_length
	return ratio



while True:

	_, frame = cap.read() #grab image from image capture
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray) # array of all the faces

		
	for face in faces:

		landmarks = predictor(gray,face)

		# get values to detect blinking
		left_eye_ratio = get_blink_ratio([36,37,38,39,40,41], landmarks)
		right_eye_ratio = get_blink_ratio([42,43,44,45,46,47], landmarks)
		blink_ratio = (left_eye_ratio + right_eye_ratio)/2

		# detect if blinking
		if blink_ratio > 6.5:
			cv2.putText(frame, "BLINK", (50,150), font, 7, (255,0,0))

		################################## Gaze detection ######################################

		left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
									(landmarks.part(37).x, landmarks.part(37).y),
									(landmarks.part(38).x, landmarks.part(38).y),
									(landmarks.part(39).x, landmarks.part(39).y),
									(landmarks.part(40).x, landmarks.part(40).y),
									(landmarks.part(41).x, landmarks.part(41).y)], np.int32)

		# Mask out only the left eye
		height, width, _ = frame.shape
		mask = np.zeros((height,width),np.uint8)
		cv2.polylines(mask, [left_eye_region], True, 255, 2)
		cv2.fillPoly(mask, [left_eye_region], 255)
		left_eye = cv2.bitwise_and(gray,gray, mask=mask)

		# segregate left eye from face
		min_x = np.min(left_eye_region[:,0])
		max_x = np.max(left_eye_region[:,0])
		min_y = np.min(left_eye_region[:,1])
		max_y = np.max(left_eye_region[:,1])

		gray_eye = left_eye[min_y:max_y, min_x:max_x]
		_, threshold_eye=cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

		height, width = threshold_eye.shape

		# get amount of white in left eye 
		left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
		left_side_white = cv2.countNonZero(left_side_threshold)
	
		right_side_threshold = threshold_eye[0:height, int(width/2):width]	
		right_side_white = cv2.countNonZero(right_side_threshold)

		cv2.putText(frame, str(left_side_white), (50,100), font, 2, (0,0,255), 3)
		cv2.putText(frame, str(right_side_white), (50,150), font, 2, (0,0,255), 3)

		threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
		eye = cv2.resize(gray_eye, None, fx=5, fy=5)

		#display trackings
		# cv2.imshow("threshold", threshold_eye)
		# cv2.imshow("left threshold", left_side_threshold)
		# cv2.imshow("right threshold", right_side_threshold)

	cv2.imshow("Frame", frame) #display image
	key = cv2.waitKey(1)
	if key == 27:
		break

# finishing up
cap.release()
cv2.destroyAllWindows()