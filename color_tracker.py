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
manual_lag_cnt = 4
 
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
        The cart only generates movement if the new and old area difference is greater than 110% or less than 90%.
        Since the target may be unstable, we do not want the cart to move constantly due to slight changes, thus we provided
        above threshold which reduce the 'noise' movement to some extent.
    '''
    if area_new > area_old * 1.1:
        move_cart_backward(area_new,area_old)
    elif area_new < area_old * 0.9:
        move_cart_forward(area_new,area_old)

"""
    since GoPiGo's camera is fixed in place, we can only turn the camera to where the cart is facing.
    We have to let the cart turn first before moving, otherwise the area of the target would be inaccurate
"""
def cart_turn(dis,ori):
    global dist_to_tar
    angle = math.atan(dis/dist_to_tar)
    degree = angle * 180 / math.pi
    if ori == 0:
        print("degree : L %f", degree)
    else:
        print("degree : R %f", degree)

"""
        since we have ao/an = (do/dn)^2, we can get that dn = sqrt(an/ao)*do
"""
def move_cart_forward(ao,an):
    global dist_to_tar
    dist_new = math.sqrt(ao/an) * dist_to_tar
    print(dist_new)
    dist_move = dist_new - dist_to_tar
    print("moveing f: %f", dist_move)
    dist_to_tar = dist_new

def move_cart_backward(ao,an):
    global dist_to_tar
    """
        since we have ao/an = (do/dn)^2, we can get that dn = sqrt(an/ao)*do
    """
    dist_new = math.sqrt(ao/an) * dist_to_tar
    print(dist_new)
    dist_move = dist_to_tar - dist_new
    dist_to_tar = math.sqrt(ao/an)
    print("moveing b: %f", dist_to_tar)
    dist_to_tar = dist_new

'''
Below function is used to find the blob and obtain the centroid of the blob
'''
center = []
"""
Input: ori_img: pre-processed (e.g. Gaussian) image
       thresh: the thresholded image you got in the previous step

"""
def detect_blob(ori_img,thresh,update):
    global target_center_new
    global target_center_old
    global area_new
    global area_old
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
    if(update == -1):
        target_center_new = center
        area_new = area
    if(update == 0):
        target_center_old = target_center_new
        target_center_new = center
        area_old = area_new
        area_new = area
        parse_cart_command()
    print(center)
    print(area)
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
    cv2.imshow('mask',mask)
    return mask

def output_cart_command():
    global target_center_new
    global target_center_old
    '''
    calculate movement based on current center and previous center if 16 frames has elapsed.
    '''


camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 4
rawCapture = PiRGBArray(camera, size=(320, 240))

''' below codes are for windows testing
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 8)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
'''
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
    cv2.namedWindow('image')
    i  = i + 1

    #if check_center == 0 it means its time to check for centroid again
    #manual_lag_cnt is provided due to significant lag of the gopigo camera. Reducing calculation time might alliviate the problem to some extent
    check_center = i % manual_lag_cnt

    key = cv2.waitKey(1)
    if key == ord("s"):
        record_flag = True
    if record_flag == True:
        list_of_clicks = getXY(image)
        x = [300,300,300]
        y = [0,0,0]
        for i in range(0,len(list_of_clicks)):
            print("hsv is:")
            print(hsv[list_of_clicks[i][1],list_of_clicks[i][0]])
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
        print("x is: ") 
        print(x)
        print("y is: ")
        print(y)
        low_thresh = np.array(x,np.uint8)
        high_thresh = np.array(y,np.uint8)
        mask = get_mask(hsv,image,low_thresh,high_thresh)
        res = cv2.bitwise_and(image,image, mask= mask)
        #pass in the -1 to indicate initialization of center and area
        res = detect_blob(image,mask,-1)
        cv2.imshow('image',res)
        record_flag = False
    else:
        if(len(low_thresh) > 0):
            #print("loading thresh")
            mask = get_mask(hsv,image,low_thresh,high_thresh)
            res = cv2.bitwise_and(image,image, mask= mask)
            blobbed = detect_blob(image,mask,check_center)
            cv2.imshow('image', blobbed)

            if(check_center == 0):
                output_cart_command()
        else:
            cv2.imshow('image',image)
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break