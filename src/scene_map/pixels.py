"""Owned generated images. Ground truth only renders and evaluates."""
import cv2
import numpy as np
from .cameras import WIDTH,HEIGHT,project

def truth_at(index):
    return {"A":np.array([-.66+.12*index,-.3,4.0]),"B":np.array([.66-.12*index,.3,6.0])}

def render(points,camera,hidden=()):
    image = np.full((HEIGHT,WIDTH,3),(28,36,40),np.uint8)
    for x in range(0,WIDTH,20): cv2.line(image,(x,0),(x,HEIGHT-1),(38,46,49),1)
    for y in range(0,HEIGHT,20): cv2.line(image,(0,y),(WIDTH-1,y),(38,46,49),1)
    for identity,point in points.items():
        if identity not in hidden:
            x,y = np.round(project(point,camera)).astype(int)
            cv2.circle(image,(int(x),int(y)),4,(155,220,235),-1)
    return image

def detect(image):
    if image.shape != (HEIGHT,WIDTH,3) or image.dtype != np.uint8:
        raise ValueError("expected320x180 uint8 image")
    mask = (image.min(axis=2)>100).astype(np.uint8)
    count,_,stats,centroids = cv2.connectedComponentsWithStats(mask,8)
    return sorted([{"x":float(centroids[i,0]),"y":float(centroids[i,1]),"area":int(stats[i,cv2.CC_STAT_AREA])} for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=20],key=lambda p:p["x"])
