"""World ground truth enters only after estimation, for scoring."""
import numpy as np
from scipy.optimize import linear_sum_assignment

class Evaluator:
    def __init__(self):
        self.errors,self.reprojection=[],[]
        self.previous_ids,self.switches,self.matched,self.skipped={},0,0,0

    def add(self,output,truth,synchronized):
        if not synchronized or not output: self.skipped+=1
        if not output: return
        identities=list(truth)
        costs=np.array([[np.linalg.norm(np.array([p["x"],p["y"],p["z"]])-truth[i]) for i in identities] for p in output])
        rows,cols=linear_sum_assignment(costs)
        for i,j in zip(rows,cols):
            identity,p=identities[j],output[i]
            self.errors.append(float(costs[i,j]**2));self.reprojection.append(p["reprojection_error_px"]);self.matched+=1
            if identity in self.previous_ids and self.previous_ids[identity]!=p["track_id"]: self.switches+=1
            self.previous_ids[identity]=p["track_id"]

    def result(self):
        return {"position_rmse_m":float(np.sqrt(np.mean(self.errors))) if self.errors else None,"mean_reprojection_error_px":float(np.mean(self.reprojection)) if self.reprojection else None,"identity_switches":self.switches,"matched_points":self.matched,"skipped_frames":self.skipped}
