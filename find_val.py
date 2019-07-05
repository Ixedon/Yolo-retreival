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
import random
from shutil import copyfile


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", default='yolo-coco',
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-s", "--size", default='big',
	help="Selcet size of test set to generate (tiny, small, medium, big)")

args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

random.seed(78)

def find_argument(text,string, indx):
	indx = text.find('<' + string + '>', indx , len(text))
	indx2 = text.find('</'+ string + '>', indx, len(text))
	arg = text[indx + 6 :indx2]

	return arg, indx2

def read_from_xml(text):
	indx = 0
	indx2 = 0

	truths = []
	while True:

		name, indx = find_argument(text,'name', indx)

		if indx == -1:
			break

		xmin, _ = find_argument(text,'xmin', indx)
		ymin, _ = find_argument(text,'ymin', indx)
		xmax, _ = find_argument(text,'xmax', indx)
		ymax, _ = find_argument(text,'ymax', indx)

		truths.append(([name, int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))], name))

	return truths

def main():

	out = open('labels-' + args['size'] +  '.txt', 'w')

	xmls = glob.glob("all/*.xml")

	k = 0 

	good_ones = []

	for xml in xmls:
		f = open(xml, 'r')

		text = f.read()

		bad = False
		print (xml)

		boxes = read_from_xml(text)
		found_all = True
		for _, name in boxes:
			found_one = False
			for label in LABELS:
				if name == label:
					found_one = True
			if not found_one:
				found_all = False
		
		pick = False
		if found_all and len(boxes) > 5:
			if args['size'] == 'big':
				pick = True
			if args['size'] == 'medium' and random.randint(0,5) == 0:
				pick = True
			if args['size'] == 'small' and random.randint(0,10) == 0:
				pick = True
			if args['size'] == 'tiny' and random.randint(0,60) == 0:
				pick = True
			if pick:
				out.write('test/' + args['size'] + '/' + xml[4:-3] + 'jpg\n')
				good_ones.append(xml)
				for box, _ in boxes:
					out.write(box[0] +' ' + str(box[1])+ ' ' + str(box[2]) + ' '+ str(box[3])+' '+ str(box[4]) + '\n')
				pick = False
				out.write(';\n')

				copyfile(xml, 'test/' + args['size'] + '/'  + xml[4:])
				copyfile(xml[0:-3] + 'jpg', 'test/' + args['size'] + '/' + xml[4:-3] + 'jpg')

		print (k)
		k += 1



	print (good_ones)
	print (len(good_ones), k)

if __name__ == "__main__":
	main()






	