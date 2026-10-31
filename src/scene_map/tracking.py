"""Constant-velocity 3D tracking with explicit observation expiry."""
import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker:
    def __init__(self,max_gap_ms=300,gate_m=1.4):
        self.max_gap_ms,self.gate_m=max_gap_ms,gate_m
        self.tracks,self.next_id,self.previous_time={},1,-1

    def update(self,observations,time_ms):
        if not np.isfinite(time_ms) or time_ms<0 or time_ms<=self.previous_time: raise ValueError("timestamps must be finite and increase")
        self.previous_time=time_ms
        self.tracks={k:t for k,t in self.tracks.items() if time_ms-t["time"]<=self.max_gap_ms}
        keys=list(self.tracks)
        assigned={}
        if keys and observations:
            costs=np.array([[np.linalg.norm(self.tracks[k]["point"]+self.tracks[k]["velocity"]*(time_ms-self.tracks[k]["time"])-o["xyz"]) for o in observations] for k in keys])
            rows,cols=linear_sum_assignment(costs)
            assigned={int(j):keys[i] for i,j in zip(rows,cols) if costs[i,j]<=self.gate_m}
        output=[]
        for i,o in enumerate(observations):
            point=o["xyz"]
            key=assigned.get(i)
            if key is None:
                key,self.next_id=self.next_id,self.next_id+1
                velocity=np.zeros(3)
            else:
                old=self.tracks[key]
                velocity=.2*(point-old["point"])/(time_ms-old["time"])+.8*old["velocity"]
            self.tracks[key]={"point":point,"velocity":velocity,"time":time_ms}
            output.append({"track_id":key,"x":float(point[0]),"y":float(point[1]),"z":float(point[2]),"reprojection_error_px":o["reprojection_error_px"]})
        return output
