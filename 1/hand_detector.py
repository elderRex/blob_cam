'''
The basic idea of this algorithm is to find the defects of every hand gesture.
Detecting different gestures based on the number of defects recognized by robot's
camera. Different number of defects correspond to different meaning, so that the
robot can move according to different hand gestures.

Uses can define specific moves by setting it up according to the number of defects.
'''

#find the convex hull of the hand
hull = cv2.convexHull(cnt)
#Find the hull
hull = cv2.convexHull(cnt,returnPoints = False)
#Draw hull in read color
cv2.drawContours(drawing,[hull],0,(0,0,255),2)
#this line can approximate polygonal curves with the specified precision
cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#get the hull array
hull = cv2.convexHull(cnt,returnPoints = False)

#Note: cnt in convexityDefects needs to be type of contour array   
if(1):
           defects = cv2.convexityDefects(cnt,hull)

           mind=0
           maxd=0
           for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                dist = cv2.pointPolygonTest(cnt,center,True)
                cv2.line(ori_img,start,end,[0,255,0],2)                 
                cv2.circle(ori_img,far,5,[0,0,255],-1)
def_num = defects.shape[0]
print def_num