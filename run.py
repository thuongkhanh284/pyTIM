import numpy as np
import numpy.matlib as npml
import math
import cv2

INPUT_FILE_NAME = 'EP03_5.avi';
OUTPUT_FILE_NAM = '';

def main():
	cap = cv2.VideoCapture(INPUT_FILE_NAME)
	num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	
    if cap.isOpened():
    	width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) )  
    	height = int( cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) )
	vector_length = width * height
	img_vector = np.zeros((vector_length , num_frame))

	# vectorize video frames 
	idx = 0
	while cap.isOpened():
		ret, img = cap.read()
		if (ret == False):
			print(" End of video ")
			break
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_vector[:,idx] = np.reshape(img_gray, vector_length)
		idx = idx + 1
	img_vector =  np.asmatrix(img_vector)
	model = TrainPGM(img_vector)

	pos = 0.5 * ()


if __name__ == '__main__':
    main()  