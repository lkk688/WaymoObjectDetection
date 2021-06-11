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

# from . import detectron2model

# initialize_model = detectron2model.initialize_model
# run_model = detectron2model.run_model
# DATA_FIELDS = detectron2model.DATA_FIELDS

Modelname="tf2od"
if Modelname=="detectron2":
    from . import detectron2model
    initialize_model = detectron2model.initialize_model
    run_model = detectron2model.run_model
    DATA_FIELDS = detectron2model.DATA_FIELDS
    setupmodeldir = detectron2model.setupmodeldir
elif Modelname=="mmdet":
    from . import mmdetmodel
    initialize_model = mmdetmodel.initialize_model
    run_model = mmdetmodel.run_model
    DATA_FIELDS = mmdetmodel.DATA_FIELDS
    setupmodeldir = mmdetmodel.setupmodeldir
elif Modelname=="tf2od":
    from . import tf2model
    initialize_model = tf2model.initialize_model
    run_model = tf2model.run_model
    DATA_FIELDS = tf2model.DATA_FIELDS
    setupmodeldir = tf2model.setupmodeldir
elif Modelname=="tf2lite":
    from . import tf2litemodel
    initialize_model = tf2litemodel.initialize_model
    run_model = tf2litemodel.run_model
    DATA_FIELDS = tf2litemodel.DATA_FIELDS
    setupmodeldir = tf2litemodel.setupmodeldir
elif Modelname=="torchvision":
    from . import torchvisionmodel
    initialize_model = torchvisionmodel.initialize_model
    run_model = torchvisionmodel.run_model
    DATA_FIELDS = torchvisionmodel.DATA_FIELDS
    setupmodeldir = torchvisionmodel.setupmodeldir