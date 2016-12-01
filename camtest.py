import io
import time
import picamera
import cv2
import numpy as np

list_of_clicks = []
def getXY(img):

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

   return list_of_clicks

stream = io.BytesIO()
camera = picamera.PiCamera()

with picamera.PiCamera() as camera:
    camera.start_preview()
    time.sleep(2)
    camera.capture(stream, format='jpeg')
# Construct a numpy array from the stream
data = np.fromstring(stream.getvalue(), dtype=np.uint8)
# "Decode" the image from the array, preserving colour
image = cv2.imdecode(data, 1)
# OpenCV returns an array with data in BGR order. If you want RGB instead
# use the following...
image = image[:, :, ::-1]

list_of_clicks = getXY(image)

print list_of_clicks