"""Example __init__.py to wrap the wod_latency_submission module imports."""
# from . import model

# initialize_model = model.initialize_model
# run_model = model.run_model
# DATA_FIELDS = model.DATA_FIELDS

# from . import mymodel

# initialize_model = mymodel.initialize_model
# run_model = mymodel.run_model
# DATA_FIELDS = mymodel.DATA_FIELDS

# from . import mmdetmodel

# initialize_model = mmdetmodel.initialize_model
# run_model = mmdetmodel.run_model
# DATA_FIELDS = mmdetmodel.DATA_FIELDS

from . import detectron2model

initialize_model = detectron2model.initialize_model
run_model = detectron2model.run_model
DATA_FIELDS = detectron2model.DATA_FIELDS