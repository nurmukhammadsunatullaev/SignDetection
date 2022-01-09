# Sign-Language-Detection
American Sign Language Recognizer

Libraries: See the code for required libraries.
___________________________________________________________________________

Train.py:

	1. Our final trained model is already provided "keras.FINAL_MODEL4".
	   Model download link: 
		https://drive.google.com/open?id=1BRhuD3msM9xKETMrRlmKAR2dqTjCTckT

	2. If model is not present, create model using this file (Needs Dataset).
		dataset:
				A:
					A_1.jpg
					..
					..
					A_200.jpg
				B:
				..
				..
				Z:
	   To run-> 'python3 train.py'
_____________________________________________________________________________

Run.py:

	1. Run using 'python run.py <index>'
		Index: 0 for default camera, 1 for external camera.
	2. Controls: 
		s	: Toggle hand band detection (green band on wrist).
			  Value can be ajusted using trackbar.
		d	: Toggle contour for debugging band detection.
		ijkl	: moving the band box in corresponding direction.
	3. Use trackbar for adjusting the HSV values for band color detection.

_____________________________________________________________________________

Create.py:

	1. Run using 'python create.py'
	2. Enter index for camera 0 or 1.
	3. i=1 for A.
	4. Press s to save the picture of hand gesture.
	
_____________________________________________________________________________
	
# SignDetection
