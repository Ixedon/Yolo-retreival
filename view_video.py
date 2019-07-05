import cv2

vs = cv2.VideoCapture('output/skufall02.avi')

while True:
	ret, frame = vs.read()

	if not ret:
		break
	if frame.shape[0] > 0:
		cv2.imshow("sd",frame)
		cv2.waitKey(0)