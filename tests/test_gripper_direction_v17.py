import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('gripper_direction_v17',Path('/workspace/grounded-vla/experiments/runpod/gripper_direction_v17.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_contact_limited_grasp_and_initial_opening_are_opposite_directions():
 q=np.zeros((3,9));a=q.copy()
 q[0,7:]=.02575;a[0,7:]=.02075
 q[1,7:]=0.;a[1,7:]=.005
 q[2,7:]=.04;a[2,7:]=.04
 np.testing.assert_array_equal(module.direction_labels(q,a),[module.CLOSE,module.OPEN,module.HOLD])
 # The old absolute-aperture threshold makes exactly the opposite grouping for first two.
 np.testing.assert_array_equal(a[:,7:].mean(1)<.015,[False,True,False])

def test_direction_invariant_to_shared_aperture_offset():
 q=np.zeros((2,9));a=q.copy();a[0,7:]=.002;a[1,7:]=-.002
 expected=module.direction_labels(q,a);q[:,7:]+=.025;a[:,7:]+=.025
 np.testing.assert_array_equal(module.direction_labels(q,a),expected)
