# Standard imports
import cv2
import numpy as np;


center = []
"""
Input: ori_img: pre-processed (e.g. Gaussian) image
	   thresh: the thresholded image you got in the previous step

"""
def detect_blob(ori_img,thresh ):
	#Find the blob by finding the contours
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	#Initialize the area and ceter position of blob
	area= 0
	center = [0,0]
	# loop over the contours
	for c in cnts:
		# compute the center of each blob
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	 	#Compute the area of each blob
	 	area_curr = cv2.contourArea(contours)
	 	if area < area_curr:
	 		area = area_curr
	 		center = (cX, cY)
	 		contour = c
	# draw the contour and center of the shape on the image
	cv2.drawContours(ori_img, [c], -1, (0, 255, 0), 2)
	cv2.circle(ori_img, center, 7, (255, 255, 255), -1)
	cv2.putText(ori_img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	 
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)