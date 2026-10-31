"""Correspondence uses image centroids, never world identities."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .cameras import triangulate

def associate(left,right,mode="geometry"):
    if mode not in ("geometry","image_order"): raise ValueError("unknown correspondence mode")
    if not left or not right: return []
    if mode=="image_order": return list(zip(sorted(left,key=lambda p:p["x"]),sorted(right,key=lambda p:p["x"])))
    costs = np.full((len(left),len(right)),1e6)
    for i,l in enumerate(left):
        for j,r in enumerate(right):
            if abs(l["y"]-r["y"])<=1.5 and 2<=l["x"]-r["x"]<=60:
                costs[i,j]=abs(l["y"]-r["y"])
    rows,cols=linear_sum_assignment(costs)
    return [(left[i],right[j]) for i,j in zip(rows,cols) if costs[i,j]<1e6]

def reconstruct(left,right,cameras,mode="geometry"):
    points=[]
    for l,r in associate(left,right,mode):
        result=triangulate([l["x"],l["y"]],[r["x"],r["y"]],*cameras)
        if result is not None:
            xyz,error=result
            if mode=="image_order" or error<=1.5: points.append({"xyz":xyz,"reprojection_error_px":error})
    return points
