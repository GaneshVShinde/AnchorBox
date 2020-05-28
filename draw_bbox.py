import cv2

fontScale = 0.5
# font = cv2.FONT_HERSHEY_SIMPLEX 
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# Blue color in BGR 
b_color = {'green':(25, 83, 217),'yellow':(32, 177, 237),'red':(142, 177, 126)}#'person':(189, 114, 0),
r_color = (0,0,255)
thickness = 1

def draw_bbox(img,x1,y1,x2,y2):
    cv2.rectangle(img,(x1,y1), (x2,y2),b_color['yellow'], 2) 
    return img
