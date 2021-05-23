# export inference model
#!python models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path {PIPELINE_CONFIG_PATH} --trained_checkpoint_dir ./myoutputmodel --output_directory ./Newexported-models2
# https://github.com/tensorflow/models/blob/master/research/object_detection/exporter_main_v2.py

import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2

tf.enable_v2_behavior()

class FLAGS:
    modelname = 'fasterrcnn_resnet50_fpn'#not used here
    #modelbasefolder = '../models/ModelZoo/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model/'
    #modelfilename='faster_rcnn_resnet50_v1_640x640_coco17_tpu-8' #not used
    #showfig='True'
    #labelmappath = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
    #threshold = 0.3
    pipeline_config_path = '/Developer/MyRepo/WaymoObjectDetection/2DObject/tfobjectdetection/tf_ssdresnet50_1024_pipeline_P100.config'
    input_type = 'image_tensor'
    trained_checkpoint_dir = '/Developer/MyRepo/mymodels/tf_ssdresnet50_output'
    output_directory='/Developer/MyRepo/mymodels/tf_ssdresnet50_output/exported'
    use_side_inputs=False
    side_input_shapes=''
    side_input_types=''
    side_input_names=''


if __name__ == '__main__':
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    #text_format.Merge(FLAGS.config_override, pipeline_config)
    exporter_lib_v2.export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_dir,
        FLAGS.output_directory, FLAGS.use_side_inputs, FLAGS.side_input_shapes,
        FLAGS.side_input_types, FLAGS.side_input_names)
