import numpy as np
import cv2
import time
import os
import shutil

def cleanup(labels):
	root = os.getcwd()
	filenames = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root,d))]
	for file in filenames:
		# Discard extras, that have a 0 in filename
		if "0" in file:
			os.remove(file)
		else:	
			# Move files to folders. These will be directly usable by the classifier
			for label in labels:
				if not os.path.exists(root+"\\" +label):
					os.makedirs(label)
				if label in file:
					shutil.move(file,label+"\\"+file)

cv2.namedWindow("Preview")
cam = cv2.VideoCapture(1)

if cam.isOpened():
	ready, frame = cam.read()
else:
	print("Camera error - close other camera instances and try again.")
	ready = False

imgBatchSize = int(input("Image batch size: "))
nlabels = int(input("No. of labels: "))
labels = ["" for x in range(nlabels)]
counter = 1

for i in range(nlabels):
	labels[i] = input("Label name: ")
	while ready and counter <= imgBatchSize:
		rval,frame = cam.read()
		key = cv2.waitKey(20)
		cv2.imshow("view", frame)
		frame = cv2.resize(frame,(240,240))
		cv2.imwrite(labels[i]+str(counter)+".jpg",frame)
		counter += 1
		if key == 27:
			break
		time.sleep(0.25)	
	counter = 0
	
cleanup(labels)
cv2.destroyWindow("preview")