# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")

args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")



def read_from_xml(text):
	
	print (text) 

	indx = 0
	indx2 = 0

	truths = []
	truths_name = []
	while True:
		indx = text.find('<name>', indx , len(text))
		indx2 = text.find('</name>', indx2, len(text))

		if indx == -1:
			#print ("end")
			break

		name = text[indx + 6 :indx2]

		indx = text.find('<xmin>', indx, len(text))
		indx2 = text.find('</xmin>', indx2 + 1, len(text))

		xmin = text[indx + 6 :indx2]

		indx = text.find('<ymin>', indx, len(text))
		indx2 = text.find('</ymin>', indx2 + 1, len(text))

		ymin = text[indx + 6 :indx2]
		

		indx = text.find('<xmax>', indx, len(text))
		indx2 = text.find('</xmax>', indx2 + 1, len(text))

		xmax = text[indx + 6 :indx2]

		indx = text.find('<ymax>', indx, len(text))
		indx2 = text.find('</ymax>', indx2 + 1, len(text))

		ymax = text[indx + 6 :indx2]

		print (name, xmin, ymin, xmax, ymax)


		truths.append([name, int(xmin), int(ymin), int(xmax), int(ymax)])
		truths_name.append(name)

	return (truths, truths_name)

out = open('labels.txt', 'w')

xmls = glob.glob("all/*.xml")

k = 0 

good_ones = []

for xml in xmls:
	f = open(xml, 'r')

	text = f.read()

	bad = False

	classes = read_from_xml(text)
	boxes, names = classes
	found_all = True
	for obj in classes:
		found_one = False
		for label in LABELS:
			if obj == label:
				found_one = True
		if not found_one:
			found_all = False
	
	if found_all and len(boxes) > 4:
		good_ones.append(xml)
		#out.write()


	print (k)
	k += 1



print (good_ones)
print (len(good_ones), k)


# for xml in good_ones:

# 	inp = open(xml, 'r')

# 	text = inp.read()

# 	out = open('Failed.py', 'w')
# 	out.write()
# 	out.close()





	