import hashlib
import threading
from pathlib import Path
from unittest.mock import patch
import pycolmap
from api import ColmapAPI

root=Path("work/sfm-real-test").resolve()
errors=[]
threading.excepthook=lambda args: errors.append(args.exc_value)

def api_for(matcher="exhaustive_matcher"):
    api=ColmapAPI(0,"OPENCV",matcher)
    api.data_path=str(root)
    return api

def run(api,recompute=False):
    errors.clear()
    api.estimate_cameras(recompute)
    api._thread.join()

def fingerprints():
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/"colmap").rglob("*") if p.is_file()}

with patch.object(pycolmap,"extract_features",side_effect=AssertionError("must use cache")):
    api=api_for()
    run(api)
    assert not errors,errors
    assert api.num_cameras>=2
    print("PASS: cache reload on fresh original-starter instance")

before=fingerprints()
with patch.object(pycolmap,"extract_features",side_effect=RuntimeError("simulated extraction failure")) as extract:
    run(api_for(),recompute=True)
    assert extract.called and len(errors)==1 and "simulated" in str(errors[0])
assert before==fingerprints()
print("PASS: forced recompute and preservation of previous cache on failure")

with patch.object(pycolmap,"extract_features",side_effect=RuntimeError("expected rebuild")) as extract:
    run(api_for("sequential_matcher"))
    assert extract.called and len(errors)==1 and "expected rebuild" in str(errors[0])
assert before==fingerprints()
print("PASS: changed GUI matcher invalidates cache")

image=next((root/"images").glob("*.jpg"))
import os
old=image.stat()
os.utime(image,ns=(old.st_atime_ns,old.st_mtime_ns+1000000000))
try:
    with patch.object(pycolmap,"extract_features",side_effect=RuntimeError("expected rebuild")) as extract:
        run(api_for())
        assert extract.called and len(errors)==1 and "expected rebuild" in str(errors[0])
finally:
    os.utime(image,ns=(old.st_atime_ns,old.st_mtime_ns))
assert before==fingerprints()
print("PASS: changed image invalidates cache")
