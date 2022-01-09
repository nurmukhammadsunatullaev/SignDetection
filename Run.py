#author: Ashutosh Upreti, Himanshu Singhvi

import cv2
import numpy as np
import tensorflow as tf
import os
import sys
from keras.models import load_model
from autocorrect import spell
model = load_model('keras.FINAL_MODEL4')

import time


image_x = 224
image_y = 224

staticBox=1
display=0

def nothing(x):
    pass

def getPosition(hsv,frame,mask):
	shape= frame.shape
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	cv2.imshow("Camera Output",mask)

	if len(cnts)>0:
		c=max(cnts,key=cv2.contourArea)
		((x,y),radius)=cv2.minEnclosingCircle(c)
		M=cv2.moments(c)

		if radius>10 :
			if display==1:
				cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),5)

			x=int(x)
			y=int(y)
			return x,y

		return (-1,-1)
	return (-1,-1)		
def pred(n):
	if n in [5,15,16]:
		return 0
	elif n in [10,21,22]:
		return 1
	elif n == 11:
		return 2
	elif n in [0,17]:
		return 3
	elif n == 9:
		return 4
	elif n == 19:
		return 5
	else:
		return 6
def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 3))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class



def recognize():
	global staticBox
	global prediction
	global display
	print(sys.argv[0])
	dvc=0
	box_x=170
	box_y=300
	cam = cv2.VideoCapture(dvc)
	blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
	greet_msg="1. For static hand detection box switch press 's'\n2. To adjust box position use ijkl.\n3. q to quit.\n"
	cv2.putText(blackboard, "S    : Kuzatuvni yoqish", (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 255))
	cv2.putText(blackboard, "IJKL : Kuzatuv sohasini boshqarish", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 255))
	cv2.putText(blackboard, "Q    : Quit", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 255))

	x, y, w, h = 300, 100, 300, 300
	old_character='a'
	text = ""
	word = ""
	count_frame = 0
	print(greet_msg)
	cv2.namedWindow('Camera Output')


	#Trackbar for adjusting the hsv values for Band detection.
	cv2.createTrackbar('H for min', 'Camera Output', 0, 255, nothing)
	cv2.createTrackbar('S for min', 'Camera Output', 0, 255, nothing)
	cv2.createTrackbar('V for min', 'Camera Output', 0, 255, nothing)
	cv2.createTrackbar('H for max', 'Camera Output', 0, 255, nothing)
	cv2.createTrackbar('S for max', 'Camera Output', 0, 255, nothing)
	cv2.createTrackbar('V for max', 'Camera Output', 0, 255, nothing)

	cv2.setTrackbarPos('H for min', 'Camera Output', 29)
	cv2.setTrackbarPos('S for min', 'Camera Output', 80)
	cv2.setTrackbarPos('V for min', 'Camera Output', 80)
	cv2.setTrackbarPos('H for max', 'Camera Output', 50)
	cv2.setTrackbarPos('S for max', 'Camera Output', 255)
	cv2.setTrackbarPos('V for max', 'Camera Output', 255)
	t = 0
	while True:
		
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		
		min_YCrCb = np.array([cv2.getTrackbarPos('H for min', 'Camera Output'),cv2.getTrackbarPos('S for min', 'Camera Output'),cv2.getTrackbarPos('V for min', 'Camera Output')], np.uint8)
		max_YCrCb = np.array([cv2.getTrackbarPos('H for max', 'Camera Output'),cv2.getTrackbarPos('S for max', 'Camera Output'),cv2.getTrackbarPos('V for max', 'Camera Output')], np.uint8)
		
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv,min_YCrCb, max_YCrCb)
		kernel = np.ones((5,5),np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


		if staticBox==1:
			c1,c2=400,400
		else:
			c1,c2=getPosition(hsv,img,mask) #returns position of Hand Band

		
		imgCrop=np.zeros((300,300,3))
		if not (c1==-1 and	c2==-1):
			x=max(0,c1-box_x)
			y=max(0,c2-box_y)
			imgCrop = img[y:y+h, x:x+w]
			imgCrop = cv2.flip(imgCrop, 1)

			labels = ['yaxshi','tinchlik','L','musht','rock and roll', 'barakalla','boshqa']

			start2=time.time()
			pred_probab, pred_class = keras_predict(model, imgCrop)
			character = labels[pred(pred_class)]
			cv2.putText(img, character, (150, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))

			if character==chr(91):
				#for none character
				cv2.putText(img, character, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
				
			else:

				if old_character == character:
					count_frame += 1
				else:
					count_frame = 0

				#if consecutive frames predict the text
				if count_frame == 8:
					if character=='Z':
						if not word=="":
							if len(word)<10:
								new_word=spell(word)
								print(new_word)
								os.system('spd-say  "'+str(new_word)+'"')
								blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
								cv2.putText(blackboard, "S    : Kuzatuvni yoqish", (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 255))
								cv2.putText(blackboard, "IJKL : Kuzatuv sohasini boshqarish", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 255))
								cv2.putText(blackboard, "Q    : Tugatish", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 255))
							
							word=""
					else:
						word = word + character +' '
						cv2.putText(blackboard, word, (80, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255))



				if len(word)>t:
					word += '\n'
					t+=20
				old_character=character
	
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
		img=np.hstack((img,blackboard))
		cv2.imshow("video",img)
	
		
		k=cv2.waitKey(50) & 0xFF
		if k== ord('s'):	#for enabling Hand Detection with Band
			staticBox=1-staticBox

		elif k == ord('q'):
			cv2.destroyAllWindows()
			print("q pressed")
			return
		# dynamic box movement adjustment ijkl
		elif k== ord('j'):
			box_x+=5
		elif k==ord('l'):
			box_x-=5
		elif k== ord('i'):
			box_y+=5
		elif k==ord('k'):
			box_y-=5

		elif k==ord('d'): #Displays contour of hand band
			display=1-display
		
		
recognize()