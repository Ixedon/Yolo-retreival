# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

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


# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
(W, H) = (None, None)


def read_from_xml():
	filename = 'test/2007_000032.xml'

	f = open(filename, 'r')

	text = f.read()
	#print (text) 

	indx = 0
	indx2 = 0

	truths = []
	while True:
		indx = text.find('<name>', indx , len(text))
		indx2 = text.find('</name>', indx2, len(text))

		if indx == -1:
			print ("end")
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

	return truths






def run_yolo(image):

	# read the next frame from the file
	# for i in range(100):
	frame = image

	(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])


	ret = []
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			ret.append([LABELS[classIDs[i]], x,y,x+w,y+h])
	
			print (LABELS[classIDs[i]], x,y,x+w,y+h)
	return ret



def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


def score(boxA, boxB):
	iou = bb_intersection_over_union(boxA[1:], boxB[1:])

	return iou


def evaluate(ground_truth, prediction):

	all_score = 0

	for truth in ground_truth:
		scores = []
		print (truth)
		for pred in prediction:
			scores.append(score (truth, pred))
		
		best_score = max(scores)
		#print (best_score)

		indx = scores.index(best_score)

		print(prediction [indx])
		print (scores)

		final_score = 0

		if truth[0] == prediction[indx][0]:
			final_score = 1
		final_score += best_score

		all_score += final_score

	print(1 - (abs (len(ground_truth) - len(prediction))/max(len(prediction), len(ground_truth))))

	all_score += (1 - (abs (len(ground_truth) - len(prediction))/max(len(prediction), len(ground_truth))))* len(ground_truth)
	print (all_score)

	return all_score/len(ground_truth)/3






image = cv2.imread('test/2007_000032.jpg')

preds = run_yolo(image)

print ("\n\n")

truths =  read_from_xml()

#score = evaluate (truths, truths)
score = evaluate (truths, preds)
print ("acc", str(int (10000 * score)/100) + "%")