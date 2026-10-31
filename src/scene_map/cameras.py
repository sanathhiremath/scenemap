"""Pinhole stereo models and linear triangulation."""
import numpy as np
WIDTH, HEIGHT, FOCAL, BASELINE = 320, 180, 150.0, 0.6

def matrices(baseline=BASELINE):
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("baseline must be finite and positive")
    k = np.array([[FOCAL,0,WIDTH/2],[0,FOCAL,HEIGHT/2],[0,0,1.]])
    return k @ np.column_stack((np.eye(3),np.zeros(3))), k @ np.column_stack((np.eye(3),[-baseline,0,0]))

def project(point, matrix):
    point = np.asarray(point,dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point must contain three finite coordinates")
    p = matrix @ np.r_[point,1.]
    if p[2] <= 0:
        raise ValueError("point must be in front of camera")
    return p[:2]/p[2]

def triangulate(left,right,camera_left,camera_right):
    left,right = np.asarray(left,float),np.asarray(right,float)
    if left.shape != (2,) or right.shape != (2,) or not np.isfinite(np.r_[left,right]).all():
        raise ValueError("image coordinates must be finite 2D points")
    a = np.vstack((left[0]*camera_left[2]-camera_left[0],left[1]*camera_left[2]-camera_left[1],right[0]*camera_right[2]-camera_right[0],right[1]*camera_right[2]-camera_right[1]))
    _,_,vt = np.linalg.svd(a)
    if abs(vt[-1,3]) < 1e-10:
        return None
    point = vt[-1,:3]/vt[-1,3]
    if point[2] <= 0 or not np.isfinite(point).all():
        return None
    error = float((np.linalg.norm(project(point,camera_left)-left)+np.linalg.norm(project(point,camera_right)-right))/2)
    return point,error
