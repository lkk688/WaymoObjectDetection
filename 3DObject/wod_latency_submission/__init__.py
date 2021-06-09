"""Example __init__.py to wrap the wod_latency_submission module imports."""
from . import mm3dmodel

initialize_model = mm3dmodel.initialize_model
run_model = mm3dmodel.run_model
DATA_FIELDS = mm3dmodel.DATA_FIELDS