import cv2 as cv
import numpy as np
import sys
from math import atan2, cos, sin, sqrt, pi
 



im = cv.imread('T_new1.png')
assert im is not None, "File could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255,0)
cnt1, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
min_area = 88000
max_area = 90000
cnt1_final = []   
for c in cnt1:
    area = cv.contourArea(c)
    #print(area)
    if min_area <= area <= max_area:
        cnt1_final.append(c)
        cv.drawContours(im, [c], -1, (0, 255, 0), 3)
#print(cnt1_final)
#print("len_of_cnt1_final",len(cnt1_final), "shape of cnt1_final: ", type(cnt1_final))
cv.imshow('T', im)
cv.waitKey(1000)
cap = cv.VideoCapture(0)





while True:
    ret, frame = cap.read()
    if not ret:
        print("The End of video")
        break
        

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_frame, (5, 5), 0)
    th3 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cnt2, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    

    #print("cnt1 analysis",cnt1)
    #print("cnt2 analysis",cnt2)
    #sys.exit()





    detection_found = False
    min_contour_area = 2000  # to eliminate that noise 
    threshold = 0.1 
#nested loop
    for c in cnt2:
        a = cv.contourArea(c)
        if a < min_contour_area:
            continue   #skips this iteration and moves on to the next iteration 
            
        for d in cnt1_final:
            ret = cv.matchShapes(d, c, 1, 0.0)
            if ret < threshold:
                detection_found = True
                
                
                rec = cv.minAreaRect(c)
                box = cv.boxPoints(rec)
                #box = np.int_(box)
                box=np.array(box,dtype=np.uint64)            
                
                # Box kai corner points kai x,y coordinates
                x1, y1 = box[0][0].item(), box[0][1].item()
                x2, y2 = box[1][0].item(), box[1][1].item()
                x3, y3 = box[2][0].item(), box[2][1].item()
                x4, y4 = box[3][0].item(), box[3][1].item()
                
               
                len1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
                len2 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2) 
                
                # humme longer line choose karna hai yaha pai 
                if len1 > len2:
                    
                    axis_start = ((x1+x4)//2, (y1+ y4)//2)
                    axis_end = ((x2+ x3)//2, (y2+ y3)//2)
                else:
                    
                    axis_start = ((x1+ x2)//2, (y1+ y2)//2)
                    axis_end = ((x4 +x3)//2, (y4+ y3)//2)
                
                
                cv.drawContours(frame, [box], 0, [255, 0, 0], 3)     #drawing blue rotating box 
                
                
                x, y, w, h = cv.boundingRect(c)
                vert_start = (x +(w//2), y)
                vert_end = (x +(w//2), y + h)
                
               
                dx = axis_end[0] - axis_start[0]
                dy = axis_end[1] - axis_start[1]
                line_angle = np.degrees(np.arctan2(dy, dx))       # tan (theta) = (y2-y1)/x2-x1
                
                # Normalize angle to always measure from vertical (90 degrees)
                #if line_angle < 0:
                    #line_angle -= 180
                    
                relative_angle = abs(90 - line_angle)
                if relative_angle > 90:
                    #relative_angle = 180 - relative_angle
                    pass
                    
                # Draw reference lines
                cv.line(frame, axis_start, axis_end, (255, 0, 0), 3)  # Reference line (taking the longer side) 
                cv.line(frame, vert_start, vert_end, (0, 0, 255), 3)  # Vertical reference (moving line)
                
                # Draw bounding box and contour
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.drawContours(frame, [c], -1, (255, 0, 0), 2)
                
                
                center_x = (x1 + x2 + x3 + x4) // 4
                center_y = (y1 + y2 + y3 + y4) // 4
                cv.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                
            
                cv.putText(frame, f"Angle: {relative_angle:.1f} deg",
                        (x - 10, y - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
                
                break
                
    
    if detection_found:
        cv.circle(frame,(50, 50), 5,(0, 255, 0), -1)  # Green circle kara print
    else:
        cv.circle(frame, (50, 50), 5,(0, 0, 255), -1)  # Red circle
    
   
    cv.imshow('Result', frame)
    
 
    if cv.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()