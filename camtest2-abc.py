# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import numpy as np
import time
import cv2
import io
import math

list_of_clicks = []
avg_threshhold = 0
record_flag = True

target_vec = []
mask = 0
low_thresh = []
high_thresh = []
 
def getXY(img):
   global record_flag

   #define the event
   def getxy_callback(event, x, y, flags, param):
    global list_of_clicks
    if event == cv2.EVENT_LBUTTONDOWN :
       list_of_clicks.append([x,y])
       print "click point is...", (x,y)

   #Read the image
   print "Reading the image..."

   #Set mouse CallBack event
   cv2.namedWindow('image')
   cv2.setMouseCallback('image', getxy_callback)

   #show the image
   print "Please select the color by clicking on the screen..."
   cv2.imshow('image', img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   #obtain the matrix of the selected points
   print "The clicked points..."
   print list_of_clicks

   record_flag = False

   return list_of_clicks

def process(hsv,image,low,high):
	global mask
	mask = cv2.inRange(hsv, low, high)
	cv2.imshow('mask',mask)
	res = cv2.bitwise_and(image,image, mask= mask)
	cv2.imshow('res',res)
	k = cv2.waitKey(5)
	res = cv2.GaussianBlur(res,(15,15),0)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
	eroded = cv2.erode(res, kernel)
	dilated = cv2.dilate(eroded,kernel)
	return dilated

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 4
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

	image = frame.array
 	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	'''
	Only initiate target setting if moust click is provoked
	If no click, simply stream the video and do the color tracking
	'''
	cv2.namedWindow('image')
	if record_flag == True:
		list_of_clicks = getXY(image)
		x = [300,300,300]
		y = [0,0,0]
		for i in range(0,len(list_of_clicks)):
			x[0] = min(x[0],image[list_of_clicks[i][1],list_of_clicks[i][0]][0])
			x[1] = min(x[0],image[list_of_clicks[i][1],list_of_clicks[i][0]][1])
			x[2] = min(x[0],image[list_of_clicks[i][1],list_of_clicks[i][0]][2])
			y[0] = max(x[0],image[list_of_clicks[i][1],list_of_clicks[i][0]][0])
			y[1] = max(x[0],image[list_of_clicks[i][1],list_of_clicks[i][0]][1])
			y[2] = max(x[0],image[list_of_clicks[i][1],list_of_clicks[i][0]][2])
		low_thresh = np.array(x)
		high_thresh = np.array(y)
		res = process(hsv,image,low_thresh, high_thresh)
		cv2.imshow('image', res)
	else:
		res = cv2.bitwise_and(image,image, mask= mask)
		cv2.imshow('image', res)

	key = cv2.waitKey(1)

	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break