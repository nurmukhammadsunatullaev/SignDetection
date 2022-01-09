import cv2
import numpy as np
import os


cam=int(input("Enter Camera Index : "))
cap=cv2.VideoCapture(cam)
i=1 # 1 for A
j=1
name=""

img3=np.zeros((640,480,3))

while(cap.isOpened()):

	_,img=cap.read()
	x=100
	y=100
	w=300
	h=300
	img1=img[y:y+h,x:x+w]
	
	img3=cv2.flip(img1,1)
	cv2.imshow("sdf",img3)
	cv2.imshow('Frame',img)
	cv2.imshow('Thresh',img1)

	#change path where you want to save files
	path="c:/"
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	#press s to save a pic	
	if k == ord('s'):
		
		directory=path+str(chr(i+64))+'/'
		if not os.path.exists(directory):
		    os.makedirs(directory)
		name=directory+str(chr(i+64))+"_"+str(j)+".jpg"
				
		cv2.imwrite(name,img1)
		if(j<200):
			j+=1
			print(j)
		else:
			for a in range(50):
				print("stop!!!!!!!!!!!!!!!!!!!!!1 ", i)
				
			j=0
			i+=1
		

cap.release()        
cv2.destroyAllWindows()