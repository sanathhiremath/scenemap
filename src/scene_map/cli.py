import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
from .cameras import WIDTH,HEIGHT,FOCAL,BASELINE,matrices
from .pixels import truth_at,render,detect
from .association import reconstruct
from .tracking import Tracker
from .evaluation import Evaluator

SCENARIOS=[("clean","Synchronized stereo"),("occlusion","A missing camera observation"),("calibration-drift","Incorrect baseline calibration"),("timestamp-skew","Reject unsynchronized camera pairs")]

def run(output):
    output.mkdir(parents=True,exist_ok=True);(output/"frames").mkdir(exist_ok=True)
    physical=matrices();scenes=[];hashes={};started=time.perf_counter()
    for scenario,title in SCENARIOS:
        baseline=BASELINE*.8 if scenario=="calibration-drift" else BASELINE
        estimated=matrices(baseline)
        trackers={m:Tracker() for m in ("geometry","image_order")};evaluators={m:Evaluator() for m in trackers};frames=[]
        for index in range(12):
            time_ms=index*100;right_time_ms=time_ms+80 if scenario=="timestamp-skew" and index>=5 else time_ms
            synchronized=abs(right_time_ms-time_ms)<=20;truth=truth_at(index)
            hidden=("A",) if scenario=="occlusion" and index in (4,5,6) else ()
            right_truth=truth_at(index+.8) if right_time_ms!=time_ms else truth
            images=(render(truth,physical[0]),render(right_truth,physical[1],hidden))
            paths=[f"frames/{scenario}-{index:02d}-{side}.png" for side in ("left","right")]
            for name,pixels in zip(paths,images):
                if not cv2.imwrite(str(output/name),pixels): raise OSError("Frame write failed")
                hashes[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
            left,right=detect(images[0]),detect(images[1]);results={}
            for mode in trackers:
                observations=reconstruct(left,right,estimated,mode) if synchronized else []
                results[mode]=trackers[mode].update(observations,time_ms)
                evaluators[mode].add(results[mode],truth,synchronized)
            frames.append({"index":index,"time_ms":time_ms,"right_time_ms":right_time_ms,"synchronized":synchronized,"left_image":paths[0],"right_image":paths[1],"detections":{"left":left,"right":right},"points":results["geometry"],"baseline_points":results["image_order"],"ground_truth":{identity:point.tolist() for identity,point in truth.items()}})
        scenes.append({"id":scenario,"title":title,"estimated_baseline_m":baseline,"frames":frames,"metrics":{m:evaluators[m].result() for m in evaluators}})
    report={"version":"0.1","name":"SceneMap","frame_size":{"width":WIDTH,"height":HEIGHT},"calibration":{"focal_px":FOCAL,"baseline_m":BASELINE,"principal_point":[WIDTH/2,HEIGHT/2],"left_projection":physical[0].tolist(),"right_projection":physical[1].tolist(),"synchronization_tolerance_ms":20},"scenarios":scenes,"summary":{"scenarios":len(scenes),"frames":48,"generated_images":96,"elapsed_seconds":time.perf_counter()-started},"provenance":{"input":"Owned rendered synchronized stereo fixtures","detector":"Bright connected-component centroids from actual image pixels","correspondence":"Epipolar row distance and positive disparity with Hungarian assignment","triangulation":"Linear DLT/SVD with positive-depth and reprojection checks","tracker":"Gated constant-velocity3D assignment","evaluation":"Generated world coordinates used only after estimation","opencv":cv2.__version__,"files_sha256":hashes},"limitations":["Owned generated pinhole stereo fixtures, not real-camera or learned-model validation.","Correspondence assumes rectified calibration and distinctive vertical positions; difficult epipolar ambiguity remains unsolved.","Identical-colored objects are detected from pixels; world identities are not estimator inputs.","Baseline scale error can leave reprojection error small while reconstructed distances are wrong.","Unsynchronized pairs produce no reconstructed points; missing observations are not silently interpolated.","Identity-switch evaluation associates estimated positions with ground truth per frame and is not a standard MOT benchmark.","Twelve frames per scenario are deterministic fixtures, not evidence of generalization."]}
    (output/"demo.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report

def main():
    parser=argparse.ArgumentParser(description="Reconstruct and track calibrated stereo observations")
    parser.add_argument("command",choices=["demo"]);parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[2]/"artifacts")
    args=parser.parse_args();report=run(args.output)
    print(json.dumps({"artifact":str(args.output/"demo.json"),"metrics":{s["id"]:s["metrics"] for s in report["scenarios"]}},indent=2))
    return 0
