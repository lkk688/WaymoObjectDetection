"""Example __init__.py to wrap the wod_latency_submission module imports."""
from . import mymodel

initialize_model = mymodel.initialize_model
run_model = mymodel.run_model
DATA_FIELDS = mymodel.DATA_FIELDS