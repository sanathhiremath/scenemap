import numpy as np
import pytest
from scene_map.cameras import matrices,project,triangulate
from scene_map.pixels import render,detect,truth_at
from scene_map.association import associate,reconstruct
from scene_map.tracking import Tracker
from scene_map.cli import run

def test_exact_calibrated_triangulation():
    cameras=matrices()
    truth=np.array([.2,.1,5.])
    result,error=triangulate(project(truth,cameras[0]),project(truth,cameras[1]),*cameras)
    assert np.allclose(result,truth,atol=1e-10)
    assert error<1e-10

def test_baseline_scale_error_is_not_detected_by_reprojection_alone():
    real=matrices(.6);estimated=matrices(.48);truth=np.array([.2,.1,5.])
    result,error=triangulate(project(truth,real[0]),project(truth,real[1]),*estimated)
    assert np.linalg.norm(result-truth)>.9
    assert error<1e-10

def test_pixel_detector_has_no_ground_truth_input():
    image=render(truth_at(0),matrices()[0])
    assert len(detect(image))==2
    image[:]=0
    assert detect(image)==[]

def test_epipolar_matching_handles_cross_camera_order_reversal():
    left=[{"x":170,"y":80},{"x":180,"y":100}]
    right=[{"x":150,"y":100},{"x":155,"y":80}]
    pairs=associate(left,right)
    assert all(a["y"]==b["y"] for a,b in pairs)
    assert any(a["y"]!=b["y"] for a,b in associate(left,right,"image_order"))

def test_missing_camera_does_not_invent_3d_observations():
    assert reconstruct([{"x":170,"y":80}],[],matrices())==[]
    tracker=Tracker()
    assert tracker.update([],0)==[]

def test_track_identity_and_timestamp_guards():
    tracker=Tracker()
    def point(x): return [{"xyz":np.array([x,0,5]),"reprojection_error_px":0.}]
    first=tracker.update(point(0),0)
    second=tracker.update(point(.1),100)
    assert first[0]["track_id"]==second[0]["track_id"]
    with pytest.raises(ValueError): tracker.update(point(.2),100)
    with pytest.raises(ValueError): tracker.update(point(.2),float('nan'))
    assert tracker.update([],200)==[]
    later=tracker.update(point(.3),500)
    assert later[0]["track_id"]!=first[0]["track_id"]

def test_reject_invalid_calibration_coordinates_and_correspondence():
    with pytest.raises(ValueError): matrices(0)
    with pytest.raises(ValueError): triangulate([np.nan,2],[1,2],*matrices())
    assert triangulate([160,90],[160,90],*matrices()) is None
    with pytest.raises(ValueError): associate([],[],"unknown")

def test_report_exposes_actual_failure_modes(tmp_path):
    report=run(tmp_path)
    scenes={s["id"]:s for s in report["scenarios"]}
    clean=scenes["clean"]["metrics"]["geometry"]
    drift=scenes["calibration-drift"]["metrics"]["geometry"]
    assert drift["position_rmse_m"]>clean["position_rmse_m"]
    assert drift["mean_reprojection_error_px"]<1e-8
    skew=scenes["timestamp-skew"]
    assert skew["metrics"]["geometry"]["skipped_frames"]==7
    assert all(not frame["points"] for frame in skew["frames"] if not frame["synchronized"])
    occluded=scenes["occlusion"]["frames"][5]
    assert len(occluded["detections"]["right"])==1
    assert len(occluded["points"])==1
    assert len(report["provenance"]["files_sha256"])==96
