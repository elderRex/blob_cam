# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import numpy as np
import time
import cv2
import io
import math
import imutils
from gopigo import *
import sys
import math

enable_encoders()

list_of_clicks = []
avg_threshhold = 0
record_flag = False

target_vec = []
mask = 0
low_thresh = []
high_thresh = []

target_center_old = []
target_center_new = []
"""
Distances are recored as in cm
Initially, we place the target 1 m away from the GoPiGo in order to provide a baseline for our calculation
"""
dist_to_tar = 100
area_old = 0
area_new = 0
fix_area = 0
manual_lag_cnt = 4

#used for naive_movement
initial_area = 0

car_axis = 11.55
r = car_axis / 2
perimeter = 20.4
 
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

'''
calculate moveing distance and degree based on current (area,center) and previous (area,center) value
With simple math, we can get:
    Diff_dist = (area_new - area_old)*dist_optical_center_to_image_plane*real_target_area / 

'''

def parse_cart_command():
    global target_center_new
    global target_center_old
    global area_new
    global area_old

    '''
    	calculate movement based on current center and previous center if 4 frames (1 second ) has elapsed.
        Get angle of movement.
        This angle shift is down prior to moving the cart
        Since the camera cannot move up&down and is restricted to left/right movement only,
        we simply calculated the difference between the coordinate of x-axis to determine the degree of turning
    '''
    dif_dis = abs(target_center_new[0] - target_center_old[0])
    move_ori = 0 if target_center_new[0] - target_center_old[0] < 0 else 1
    dif_vec = [dif_dis, move_ori]
    #if new&old center difference is greater than 20pixel, then we assume it's noticeable angle change and cart need to adjust accordingly
    if dif_dis > 20:
        cart_turn(dif_vec[0],dif_vec[1])

    '''
        Get distance based on area change
        The cart only generates movement if the new and old area difference is greater than 105% or less than 95%.
        Since the target may be unstable, we do not want the cart to move constantly due to slight changes, thus we provided
        above threshold which reduce the 'noise' movement to some extent.
    '''
    if area_new > fix_area * 1.05:
        move_cart_backward(area_new,area_old)
    elif area_new < fix_area * 0.95:
        move_cart_forward(area_new,area_old)

"""
    since GoPiGo's camera is fixed in place, we can only turn the camera to where the cart is facing.
    We have to let the cart turn first before moving, otherwise the area of the target would be inaccurate
"""
def cart_turn(dis,ori):
    global dist_to_tar
    angle = math.atan(dis/dist_to_tar)
    degree = int(angle * 180 / math.pi)
    target = int((angle * 3.14 / 180) * r *18/ perimeter)
    print "target"+str(target)

    if ori == 0:
        print("degree : L %f", degree)
        rotate_itselfL(abs(target))
    else:
        print("degree : R %f", degree)
        rotate_itselfR(abs(target))

def rotate_itselfR(target):
    enc_tgt(1, 1, target)
    set_speed(180)
    right_rot()

def rotate_itselfL(target):
    enc_tgt(1, 1, target)
    set_speed(180)
    left_rot()

"""
        since we have ao/an = (do/dn)^2, we can get that dn = sqrt(an/ao)*do
        target here is used for movement
"""
def move_cart_forward(ao,an):
	global dist_to_tar
	global perimeter
	dist_new = math.sqrt(ao/an) * dist_to_tar
	print(dist_new)
	dist_move = dist_new - dist_to_tar
	print("moveing f: %f", dist_move)
	dist_to_tar = dist_new
	target = int(dist_new/perimeter)
	enc_tgt(1,1,target)    
	set_speed(180)
	fwd()


def move_cart_backward(ao,an):
	global dist_to_tar
	global perimeter
	dist_new = math.sqrt(ao/an) * dist_to_tar
	print(dist_new)
	dist_move = dist_to_tar - dist_new
	dist_to_tar = math.sqrt(ao/an)
	print("moveing b: %f", dist_to_tar)
	dist_to_tar = dist_new
	target = int(dist_new/perimeter)
	enc_tgt(1,1,target)     
	set_speed(180)
	bwd()

'''
Below function is used to find the blob and obtain the centroid of the blob
'''
center = []
"""
Input: ori_img: pre-processed (e.g. Gaussian) image
       thresh: the thresholded image you got in the previous step

"""
def detect_blob(ori_img,thresh,ini):
    global target_center_new
    global target_center_old
    global area_new
    global area_old
    global fix_area
    #Find the blob by finding the contours
    img,cnts,hierachy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(cnts)
    #print("------")

    #Initialize the area and ceter position of blob
    area= 0
    cX = 0
    cY = 0
    center = (0,0)
    large_contour = []
    # loop over the contours
    for c in cnts:
        # compute the center of each blob
        M = cv2.moments(c)
        #print(M["m00"])
        if(M['m00']!=0):
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX = 0
            cY = 0
        #Compute the area of each blob
        area_curr = cv2.contourArea(c)
        if area < area_curr:
            area = area_curr
            center = (cX, cY)
            large_contour = c
    if ini == 1:
        target_center_new = center
        area_new = area
        fix_area = area
        '''
        	#used for Method 1 - Naive Movement
        initial_area = area
        '''
    else:
        target_center_old = target_center_new
        target_center_new = center
        area_old = area_new
        area_new = area
        parse_cart_command()
        #naive_movement()
    print(center)
    print(area)
    #maker = cv2.minAreaRect(large_contour)
    #print(maker[1][0])
    #cv2.waitKey(0)
    # draw the contour and center of the shape on the image
    cv2.drawContours(ori_img, large_contour, -1, (0, 255, 0), 3)
    cv2.circle(ori_img, center, 7, (0, 0, 255), -1)
    cv2.putText(ori_img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # show the image
    return ori_img



def get_mask(hsv,image,low,high):
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.GaussianBlur(mask,(15,15),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
    eroded = cv2.erode(mask, kernel)
    mask = cv2.dilate(eroded,kernel)
    return mask

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 4
rawCapture = PiRGBArray(camera, size=(320, 240))
 
# allow the camera to warmup
time.sleep(0.1)

'''
    Step1. Read the each frame
    Step2. If 's' key is pressed, it indicates a color choice session need to started. If so, switch to GetXY
    Step3. After image has been processed according to the requirement, find the blob
    Step4. Get blob center and size - biggest blob is always the target
    Step5. Generate Movement commands accordingly
    Step6. Pass the commands to GoPiGO
    Step7. Re-evalute the situation
'''
i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

	image = frame.array
 	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	'''
	Only initiate target setting if moust click is provoked
	If no click, simply stream the video and do the color tracking
	'''
	cv2.namedWindow('image')
	i = i + 1
	#if check_center == 0 it means its time to check for centroid again

	if record_flag == True:
		list_of_clicks = getXY(image)
		x = [300,300,300]
		y = [0,0,0]
		for i in range(0,len(list_of_clicks)):
			if(y[0] < hsv[list_of_clicks[i][1],list_of_clicks[i][0]][0]):
				y[0] = hsv[list_of_clicks[i][1],list_of_clicks[i][0]][0]
				#y[0] = 5
				y[1] = 255
				y[2] = 255
			if(x[0] >= hsv[list_of_clicks[i][1],list_of_clicks[i][0]][0]):
				x[0] = hsv[list_of_clicks[i][1],list_of_clicks[i][0]][0]
                #x[0] = 1
				x[1] = 50
				x[2] = 50
		low_thresh = np.array(x,np.uint8)
		high_thresh = np.array(y,np.uint8)
		mask = get_mask(hsv,image,low_thresh,high_thresh)
		res = cv2.bitwise_and(image,image, mask= mask)
		#pass in the -1 to indicate initialization of center and area
		res = detect_blob(image,mask,1)
		cv2.imshow('image',res)
		record_flag = False
	else:
		if(len(low_thresh) > 0):
 			#print("loading thresh")
			mask = get_mask(hsv,image,low_thresh,high_thresh)
			res = cv2.bitwise_and(image,image, mask= mask)
			blobbed = detect_blob(image,mask,0)
			cv2.imshow('image', blobbed)
			'''
				Movement detection Method 1 - (README tag: Naive_Movement)
				Below method simply moves the cart the checks the new_area with the initial area
				Since area_new is updated in every frame, we can obatin rather accurate readings of it in the graph
				In order to reduce the noise and accidents,
				we simple required the cart to move at a given length (encoder count 2) each time and re-compare the results.
				The reason we implemented this is because of the controlling issue with the cart.
			'''
			'''
			if area_new > initial_area * 1.05:
		        enc_tgt(1,1,2)    
				set_speed(180)
				fwd()
		    elif area_new < initial_area * 0.95:
		        enc_tgt(1,1,2)    
				set_speed(180)
				bwd()
			'''
		else:
			cv2.imshow('image',image)

	rawCapture.truncate(0)
 
	key = cv2.waitKey(1)
	if key == ord("s"):
		record_flag = True