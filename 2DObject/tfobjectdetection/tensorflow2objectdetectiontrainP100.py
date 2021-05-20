from object_detection.core import standard_fields as fields
from object_detection.utils import ops
from object_detection import model_lib
from object_detection.protos import train_pb2
import math
import sys
import time
import os
from object_detection import model_lib_v2
import tensorflow.compat.v2 as tf
from absl import flags
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import cv2
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import numpy as np
import scipy.misc
import io
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())

print("Tensorflow Version: ", tf.__version__)
print("Keras Version: ", tf.keras.__version__)

# check GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)


# import tensorflow as tf


# ref: https://github.com/tensorflow/models/blob/master/official/vision/detection/dataloader/tf_example_decoder.py

def _decode_image(parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    return image


def _decode_boxes(parsed_tensors):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    print(ymax)

    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def _decode_areas(parsed_tensors):
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    return tf.cond(
        tf.greater(tf.shape(parsed_tensors['image/object/area'])[0], 0),
        lambda: parsed_tensors['image/object/area'],
        lambda: (xmax - xmin) * (ymax - ymin))


# classlabelkeyname='image/object/class/label' #used in the previous TF record file
classlabelkeyname = 'image/object/class/text'  # used in the new TF record file
# 'image/object/class/text':
#             tf.io.VarLenFeature(tf.int64),


def read_tfrecord(example):
    features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string),
        'image/source_id':
            tf.io.FixedLenFeature((), tf.string),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.io.VarLenFeature(tf.string),
        'image/object/area':
            tf.io.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.io.VarLenFeature(tf.int64),
    }
    # features = {
    #     "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    #     "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    # }
    example = tf.io.parse_single_example(example, features)

    for k in example:
        if isinstance(example[k], tf.SparseTensor):
            if example[k].dtype == tf.string:
                example[k] = tf.sparse.to_dense(
                    example[k], default_value='')
            else:
                example[k] = tf.sparse.to_dense(
                    example[k], default_value=0)

    print("Got example")
    print(example['image/object/bbox/xmin'])
    image = _decode_image(example)
    print("Decoded image")
    boxes = _decode_boxes(example)
    print("Decoded boxes:", boxes)
    areas = _decode_areas(example)
    is_crowds = tf.cond(
        tf.greater(tf.shape(example['image/object/is_crowd'])[0], 0),
        lambda: tf.cast(example['image/object/is_crowd'], dtype=tf.bool),
        lambda: tf.zeros_like(example['image/object/class/label'], dtype=tf.bool))  # pylint: disable=line-too-long

    source_id = example['image/source_id']
    height = example['image/height']
    width = example['image/width']
    # ['image/object/class/label']
    groundtruth_class = example['image/object/class/label']

    # image = tf.image.decode_jpeg(example['image'], channels=3)
    # convert image to floats in [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)
    # image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size will be needed for TPU
    # class_label = example['class']
    print(groundtruth_class)
    decoded_tensors = {
        'image': image,
        'source_id': source_id,
        'height': height,
        'width': width,
        'groundtruth_classes': groundtruth_class,
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    return decoded_tensors  # image, class_label


def load_dataset(filenames):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset


INSTANCE_CATEGORY_NAMES = ['unknown',
                           'vehicle', 'pedestrian', 'sign', 'cyclist']
# INSTANCE_Color = {
#     'Unknown':'black', b'vehicle':'red', b'pedestrian':'green', b'sign':'red', b'cyclist':'purple'
# }#'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
INSTANCE_Color = ['black', 'red', 'green', 'red', 'purple']


def show_oneimage_category(image, label, boundingbox, IMAGE_SIZE):
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1)

    len = label.size
    print(len)
    img_height = IMAGE_SIZE[0]
    img_width = IMAGE_SIZE[1]
    for index in range(len):
        box = boundingbox[index]
        labelid = label[index]
        print(labelid)
        # labelid #INSTANCE_CATEGORY_NAMES[labelid]
        labelname = INSTANCE_CATEGORY_NAMES[labelid]
        classcolor = INSTANCE_Color[int(labelid)]  # labelname]
        # [xmin, ymin, xmax, ymax]=box#*IMAGE_SIZE[0]
        [ymin, xmin, ymax, xmax] = box
        xmin = xmin*img_width
        xmax = xmax*img_width
        ymin = ymin*img_height
        ymax = ymax*img_height
        boxwidth = xmax-xmin
        boxheight = ymax-ymin
        if (boxwidth/img_width > 0.01 and boxheight/img_height > 0.01):
            print("Class id:", labelid)
            print(box)
            startpoint = (xmin, ymin)
            end_point = (xmax, ymax)
            # cv2.rectangle(image, startpoint, end_point, color=(0, 255, 0), thickness=1) # Draw Rectangle with the coordinates
            # Draw the object bounding box. https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
            ax.add_patch(patches.Rectangle(
                xy=(xmin,
                    ymin),
                width=boxwidth,
                height=boxheight,
                linewidth=1,
                edgecolor=classcolor,  # 'red',
                facecolor='none'))
            # ax.add_patch(patches.Rectangle(
            #     xy=(ymin,
            #         xmin),
            #     width=ymax-ymin,
            #     height=xmax-xmin,
            #     linewidth=1,
            #     edgecolor=classcolor, #'red',
            #     facecolor='none'))
            # ax.text(ymin, xmin, labelname, color=classcolor, fontsize=10)
            ax.text(xmin, ymin, labelname, color=classcolor, fontsize=10)
            text_size = 1
            # cv2.putText(image, labelname, startpoint,  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=1)

    # return img
    # plt.imshow(image)
    plt.imsave('test.png', image)
    # plt.title(CLASSES[label_batch.numpy()])


def is_object_based_checkpoint(checkpoint_path):
    """Returns true if `checkpoint_path` points to an object-based checkpoint."""
    var_names = [var[0] for var in tf.train.list_variables(checkpoint_path)]
    return '_CHECKPOINTABLE_OBJECT_GRAPH' in var_names


def _compute_losses_and_predictions_dicts(
        model, features, labels,
        add_regularization_loss=True):
    """Computes the losses dict and predictions dict for a model on inputs.
    Args:
      model: a DetectionModel (based on Keras).
      features: Dictionary of feature tensors from the input dataset.
        Should be in the format output by `inputs.train_input` and
        `inputs.eval_input`.
          features[fields.InputDataFields.image] is a [batch_size, H, W, C]
            float32 tensor with preprocessed images.
          features[HASH_KEY] is a [batch_size] int32 tensor representing unique
            identifiers for the images.
          features[fields.InputDataFields.true_image_shape] is a [
              batch_size, 3]
            int32 tensor representing the true image shapes, as preprocessed
            images could be padded.
          features[fields.InputDataFields.original_image] (optional) is a
            [batch_size, H, W, C] float32 tensor with original images.
      labels: A dictionary of groundtruth tensors post-unstacking. The original
        labels are of the form returned by `inputs.train_input` and
        `inputs.eval_input`. The shapes may have been modified by unstacking with
        `model_lib.unstack_batch`. However, the dictionary includes the following
        fields.
          labels[fields.InputDataFields.num_groundtruth_boxes] is a
            int32 tensor indicating the number of valid groundtruth boxes
            per image.
          labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor
            containing the corners of the groundtruth boxes.
          labels[fields.InputDataFields.groundtruth_classes] is a float32
            one-hot tensor of classes.
          labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor
            containing groundtruth weights for the boxes.
          -- Optional --
          labels[fields.InputDataFields.groundtruth_instance_masks] is a
            float32 tensor containing only binary values, which represent
            instance masks for objects.
          labels[fields.InputDataFields.groundtruth_keypoints] is a
            float32 tensor containing keypoints for each box.
          labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32
            tensor with the number of sampled DensePose points per object.
          labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32
            tensor with the DensePose part ids (0-indexed) per object.
          labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
            float32 tensor with the DensePose surface coordinates.
          labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor
            containing group_of annotations.
          labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
            k-hot tensor of classes.
          labels[fields.InputDataFields.groundtruth_track_ids] is a int32
            tensor of track IDs.
          labels[fields.InputDataFields.groundtruth_keypoint_depths] is a
            float32 tensor containing keypoint depths information.
          labels[fields.InputDataFields.groundtruth_keypoint_depth_weights] is a
            float32 tensor containing the weights of the keypoint depth feature.
      add_regularization_loss: Whether or not to include the model's
        regularization loss in the losses dictionary.
    Returns:
      A tuple containing the losses dictionary (with the total loss under
      the key 'Loss/total_loss'), and the predictions dictionary produced by
      `model.predict`.
    """
    model_lib.provide_groundtruth(model, labels)
    preprocessed_images = features[fields.InputDataFields.image]

    prediction_dict = model.predict(
        preprocessed_images,
        features[fields.InputDataFields.true_image_shape],
        **model.get_side_inputs(features))
    prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

    losses_dict = model.loss(
        prediction_dict, features[fields.InputDataFields.true_image_shape])
    losses = [loss_tensor for loss_tensor in losses_dict.values()]
    if add_regularization_loss:
        # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
        # need to convert these regularization losses from bfloat16 to float32
        # as well.
        regularization_losses = model.regularization_losses()
        if regularization_losses:
            regularization_losses = ops.bfloat16_to_float32_nested(
                regularization_losses)
            regularization_loss = tf.add_n(
                regularization_losses, name='regularization_loss')
            losses.append(regularization_loss)
            losses_dict['Loss/regularization_loss'] = regularization_loss

    total_loss = tf.add_n(losses, name='total_loss')
    losses_dict['Loss/total_loss'] = total_loss

    return losses_dict, prediction_dict


def _ensure_model_is_built(model, input_dataset, unpad_groundtruth_tensors):
    """Ensures that model variables are all built, by running on a dummy input.
    Args:
      model: A DetectionModel to be built.
      input_dataset: The tf.data Dataset the model is being trained on. Needed to
        get the shapes for the dummy loss computation.
      unpad_groundtruth_tensors: A parameter passed to unstack_batch.
    """
    features, labels = iter(input_dataset).next()

    @tf.function
    def _dummy_computation_fn(features, labels):
        model._is_training = False  # pylint: disable=protected-access
        tf.keras.backend.set_learning_phase(False)

        labels = model_lib.unstack_batch(
            labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

        return _compute_losses_and_predictions_dicts(model, features, labels)

    strategy = tf.compat.v2.distribute.get_strategy()
    if hasattr(tf.distribute.Strategy, 'run'):
        strategy.run(
            _dummy_computation_fn, args=(
                features,
                labels,
            ))
    else:
        strategy.experimental_run_v2(
            _dummy_computation_fn, args=(
                features,
                labels,
            ))


RESTORE_MAP_ERROR_TEMPLATE = (
    'Since we are restoring a v2 style checkpoint'
    ' restore_map was expected to return a (str -> Model) mapping,'
    ' but we received a ({} -> {}) mapping instead.'
)


def validate_tf_v2_checkpoint_restore_map(checkpoint_restore_map):
    """Ensure that given dict is a valid TF v2 style restore map.
    Args:
      checkpoint_restore_map: A nested dict mapping strings to
        tf.keras.Model objects.
    Raises:
      ValueError: If they keys in checkpoint_restore_map are not strings or if
        the values are not keras Model objects.
    """

    for key, value in checkpoint_restore_map.items():
        if not (isinstance(key, str) and
                (isinstance(value, tf.Module)
                 or isinstance(value, tf.train.Checkpoint))):
            if isinstance(key, str) and isinstance(value, dict):
                validate_tf_v2_checkpoint_restore_map(value)
            else:
                raise TypeError(
                    RESTORE_MAP_ERROR_TEMPLATE.format(key.__class__.__name__,
                                                      value.__class__.__name__))


def load_fine_tune_checkpoint(model, checkpoint_path, checkpoint_type,
                              checkpoint_version, run_model_on_dummy_input,
                              input_dataset, unpad_groundtruth_tensors):
    """Load a fine tuning classification or detection checkpoint.
    To make sure the model variables are all built, this method first executes
    the model by computing a dummy loss. (Models might not have built their
    variables before their first execution)
    It then loads an object-based classification or detection checkpoint.
    This method updates the model in-place and does not return a value.
    Args:
      model: A DetectionModel (based on Keras) to load a fine-tuning
        checkpoint for.
      checkpoint_path: Directory with checkpoints file or path to checkpoint.
      checkpoint_type: Whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`.
      checkpoint_version: train_pb2.CheckpointVersion.V1 or V2 enum indicating
        whether to load checkpoints in V1 style or V2 style.  In this binary
        we only support V2 style (object-based) checkpoints.
      run_model_on_dummy_input: Whether to run the model on a dummy input in order
        to ensure that all model variables have been built successfully before
        loading the fine_tune_checkpoint.
      input_dataset: The tf.data Dataset the model is being trained on. Needed
        to get the shapes for the dummy loss computation.
      unpad_groundtruth_tensors: A parameter passed to unstack_batch.
    Raises:
      IOError: if `checkpoint_path` does not point at a valid object-based
        checkpoint
      ValueError: if `checkpoint_version` is not train_pb2.CheckpointVersion.V2
    """
    if not is_object_based_checkpoint(checkpoint_path):
        raise IOError(
            'Checkpoint is expected to be an object-based checkpoint.')
    if checkpoint_version == train_pb2.CheckpointVersion.V1:
        raise ValueError('Checkpoint version should be V2')

    if run_model_on_dummy_input:
        _ensure_model_is_built(model, input_dataset, unpad_groundtruth_tensors)

    restore_from_objects_dict = model.restore_from_objects(
        fine_tune_checkpoint_type=checkpoint_type)
    validate_tf_v2_checkpoint_restore_map(restore_from_objects_dict)
    ckpt = tf.train.Checkpoint(**restore_from_objects_dict)
    ckpt.restore(checkpoint_path).assert_existing_objects_matched()


def get_filepath(strategy, filepath):
    """Get appropriate filepath for worker.
    Args:
      strategy: A tf.distribute.Strategy object.
      filepath: A path to where the Checkpoint object is stored.
    Returns:
      A temporary filepath for non-chief workers to use or the original filepath
      for the chief.
    """
    if strategy.extended.should_checkpoint:
        return filepath
    else:
        # TODO(vighneshb) Replace with the public API when TF exposes it.
        task_id = strategy.extended._task_id  # pylint:disable=protected-access
        return os.path.join(filepath, 'temp_worker_{:03d}'.format(task_id))


def eager_train_step(detection_model,
                     features,
                     labels,
                     unpad_groundtruth_tensors,
                     optimizer,
                     learning_rate,
                     add_regularization_loss=True,
                     clip_gradients_value=None,
                     global_step=None,
                     num_replicas=1.0):
    """Process a single training batch.
    This method computes the loss for the model on a single training batch,
    while tracking the gradients with a gradient tape. It then updates the
    model variables with the optimizer, clipping the gradients if
    clip_gradients_value is present.
    This method can run eagerly or inside a tf.function.
    Args:
      detection_model: A DetectionModel (based on Keras) to train.
      features: Dictionary of feature tensors from the input dataset.
        Should be in the format output by `inputs.train_input.
          features[fields.InputDataFields.image] is a [batch_size, H, W, C]
            float32 tensor with preprocessed images.
          features[HASH_KEY] is a [batch_size] int32 tensor representing unique
            identifiers for the images.
          features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
            int32 tensor representing the true image shapes, as preprocessed
            images could be padded.
          features[fields.InputDataFields.original_image] (optional, not used
            during training) is a
            [batch_size, H, W, C] float32 tensor with original images.
      labels: A dictionary of groundtruth tensors. This method unstacks
        these labels using model_lib.unstack_batch. The stacked labels are of
        the form returned by `inputs.train_input` and `inputs.eval_input`.
          labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
            int32 tensor indicating the number of valid groundtruth boxes
            per image.
          labels[fields.InputDataFields.groundtruth_boxes] is a
            [batch_size, num_boxes, 4] float32 tensor containing the corners of
            the groundtruth boxes.
          labels[fields.InputDataFields.groundtruth_classes] is a
            [batch_size, num_boxes, num_classes] float32 one-hot tensor of
            classes. num_classes includes the background class.
          labels[fields.InputDataFields.groundtruth_weights] is a
            [batch_size, num_boxes] float32 tensor containing groundtruth weights
            for the boxes.
          -- Optional --
          labels[fields.InputDataFields.groundtruth_instance_masks] is a
            [batch_size, num_boxes, H, W] float32 tensor containing only binary
            values, which represent instance masks for objects.
          labels[fields.InputDataFields.groundtruth_keypoints] is a
            [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
            keypoints for each box.
          labels[fields.InputDataFields.groundtruth_dp_num_points] is a
            [batch_size, num_boxes] int32 tensor with the number of DensePose
            sampled points per instance.
          labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
            [batch_size, num_boxes, max_sampled_points] int32 tensor with the
            part ids (0-indexed) for each instance.
          labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
            [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
            surface coordinates for each point. Each surface coordinate is of the
            form (y, x, v, u) where (y, x) are normalized image locations and
            (v, u) are part-relative normalized surface coordinates.
          labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
            k-hot tensor of classes.
          labels[fields.InputDataFields.groundtruth_track_ids] is a int32
            tensor of track IDs.
          labels[fields.InputDataFields.groundtruth_keypoint_depths] is a
            float32 tensor containing keypoint depths information.
          labels[fields.InputDataFields.groundtruth_keypoint_depth_weights] is a
            float32 tensor containing the weights of the keypoint depth feature.
      unpad_groundtruth_tensors: A parameter passed to unstack_batch.
      optimizer: The training optimizer that will update the variables.
      learning_rate: The learning rate tensor for the current training step.
        This is used only for TensorBoard logging purposes, it does not affect
         model training.
      add_regularization_loss: Whether or not to include the model's
        regularization loss in the losses dictionary.
      clip_gradients_value: If this is present, clip the gradients global norm
        at this value using `tf.clip_by_global_norm`.
      global_step: The current training step. Used for TensorBoard logging
        purposes. This step is not updated by this function and must be
        incremented separately.
      num_replicas: The number of replicas in the current distribution strategy.
        This is used to scale the total loss so that training in a distribution
        strategy works correctly.
    Returns:
      The total loss observed at this training step
    """
    # """Execute a single training step in the TF v2 style loop."""
    is_training = True

    detection_model._is_training = is_training  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(is_training)

    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    with tf.GradientTape() as tape:
        losses_dict, _ = _compute_losses_and_predictions_dicts(
            detection_model, features, labels, add_regularization_loss)

        total_loss = losses_dict['Loss/total_loss']

        # Normalize loss for num replicas
        total_loss = tf.math.divide(total_loss,
                                    tf.constant(num_replicas, dtype=tf.float32))
        losses_dict['Loss/normalized_total_loss'] = total_loss

    for loss_type in losses_dict:
        tf.compat.v2.summary.scalar(
            loss_type, losses_dict[loss_type], step=global_step)

    trainable_variables = detection_model.trainable_variables

    gradients = tape.gradient(total_loss, trainable_variables)

    if clip_gradients_value:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients_value)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    tf.compat.v2.summary.scalar(
        'learning_rate', learning_rate, step=global_step)
    tf.compat.v2.summary.image(
        name='train_input_images',
        step=global_step,
        data=features[fields.InputDataFields.image],
        max_outputs=3)
    return total_loss


NUM_STEPS_PER_ITERATION = 100


def train_loop(
        pipeline_config_path,
        model_dir,
        config_override=None,
        train_steps=None,
        use_tpu=False,
        save_final_config=False,
        checkpoint_every_n=1000,
        checkpoint_max_to_keep=7,
        record_summaries=True,
        performance_summary_exporter=None,
        num_steps_per_iteration=NUM_STEPS_PER_ITERATION,
        **kwargs):
    # """Trains a model using eager + functions.

    config_override = None
    configs = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override)
    kwargs.update({
        'train_steps': train_steps,
        'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu
    })
    configs = config_util.merge_external_params_with_configs(
        configs, None, kwargs_dict=kwargs)
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors  # False
    add_regularization_loss = train_config.add_regularization_loss  # True
    clip_gradients_value = None
    if train_config.gradient_clipping_by_norm > 0:  # Not run
        clip_gradients_value = train_config.gradient_clipping_by_norm

    # update train_steps from config but only when non-zero value is provided
    train_steps = num_train_steps
    if train_steps is None and train_config.num_steps != 0:
        train_steps = train_config.num_steps

    tf.compat.v2.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    if train_config.load_all_detection_checkpoint_vars:
        raise ValueError('train_pb2.load_all_detection_checkpoint_vars '
                         'unsupported in TF2')

    config_util.update_fine_tune_checkpoint_type(train_config)
    fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type  # 'detection'
    fine_tune_checkpoint_version = train_config.fine_tune_checkpoint_version

    # Build the model, optimizer, and training input
    strategy = tf.compat.v2.distribute.get_strategy()
    from object_detection import inputs
    from object_detection.builders import optimizer_builder
    from object_detection.utils import variables_helper
    with strategy.scope():
        detection_model = model_builder.build(
            model_config=model_config, is_training=True)

        def train_dataset_fn(input_context):
            """Callable to create train input."""
            # Create the inputs.
            train_input = inputs.train_input(
                train_config=train_config,
                train_input_config=train_input_config,
                model_config=model_config,
                model=detection_model,
                input_context=input_context)
            train_input = train_input.repeat()
            return train_input

        train_input = strategy.experimental_distribute_datasets_from_function(
            train_dataset_fn)
        global_step = tf.Variable(0, trainable=False, dtype=tf.compat.v2.dtypes.int64,
                                  name='global_step', aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)
        optimizer, (learning_rate,) = optimizer_builder.build(
            train_config.optimizer, global_step=global_step)

        if callable(learning_rate):
            learning_rate_fn = learning_rate
        else:
            def learning_rate_fn(): return learning_rate

    # Train the model
    # Get the appropriate filepath (temporary or not) based on whether the worker
    # is the chief.
    summary_writer_filepath = get_filepath(strategy,
                                           os.path.join(model_dir, 'train'))
    if record_summaries:
        summary_writer = tf.compat.v2.summary.create_file_writer(
            summary_writer_filepath)
    else:
        #summary_writer = tf2.summary.create_noop_writer()
        summary_writer = tf.summary.create_noop_writer()

    with summary_writer.as_default():
        with strategy.scope():
            with tf.compat.v2.summary.record_if(
                    lambda: global_step % num_steps_per_iteration == 0):
                # Load a fine-tuning checkpoint.
                if train_config.fine_tune_checkpoint:
                    variables_helper.ensure_checkpoint_supported(
                        train_config.fine_tune_checkpoint, fine_tune_checkpoint_type,
                        model_dir)
                    load_fine_tune_checkpoint(
                        detection_model, train_config.fine_tune_checkpoint,
                        fine_tune_checkpoint_type, fine_tune_checkpoint_version,
                        train_config.run_fine_tune_checkpoint_dummy_computation,
                        train_input, unpad_groundtruth_tensors)

                ckpt = tf.compat.v2.train.Checkpoint(
                    step=global_step, model=detection_model, optimizer=optimizer)

                manager_dir = get_filepath(strategy, model_dir)
                if not strategy.extended.should_checkpoint:
                    checkpoint_max_to_keep = 1
                manager = tf.compat.v2.train.CheckpointManager(
                    ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep)

                # We use the following instead of manager.latest_checkpoint because
                # manager_dir does not point to the model directory when we are running
                # in a worker.
                latest_checkpoint = tf.train.latest_checkpoint(model_dir)
                ckpt.restore(latest_checkpoint)

                def train_step_fn(features, labels):
                    """Single train step."""
                    loss = eager_train_step(
                        detection_model,
                        features,
                        labels,
                        unpad_groundtruth_tensors,
                        optimizer,
                        learning_rate=learning_rate_fn(),
                        add_regularization_loss=add_regularization_loss,
                        clip_gradients_value=clip_gradients_value,
                        global_step=global_step,
                        num_replicas=strategy.num_replicas_in_sync)

                def _sample_and_train(strategy, train_step_fn, data_iterator):
                    features, labels = data_iterator.next()
                    if hasattr(tf.distribute.Strategy, 'run'):
                        per_replica_losses = strategy.run(
                            train_step_fn, args=(features, labels))
                    else:
                        per_replica_losses = strategy.experimental_run_v2(
                            train_step_fn, args=(features, labels))
                    # TODO(anjalisridhar): explore if it is safe to remove the
                    # num_replicas scaling of the loss and switch this to a ReduceOp.Mean
                    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                                        per_replica_losses, axis=None)

                @tf.function
                def _dist_train_step(data_iterator):
                    """A distributed train step."""

                    if num_steps_per_iteration > 1:
                        for _ in tf.range(num_steps_per_iteration - 1):
                            # Following suggestion on yaqs/5402607292645376
                            with tf.name_scope(''):
                                _sample_and_train(
                                    strategy, train_step_fn, data_iterator)

                    return _sample_and_train(strategy, train_step_fn, data_iterator)

                train_input_iter = iter(train_input)

                if int(global_step.value()) == 0:
                    manager.save()

                checkpointed_step = int(global_step.value())
                logged_step = global_step.value()

                last_step_time = time.time()
                for _ in range(global_step.value(), train_steps,
                            num_steps_per_iteration):

                    loss = _dist_train_step(train_input_iter)

                    time_taken = time.time() - last_step_time
                    last_step_time = time.time()
                    steps_per_sec = num_steps_per_iteration * 1.0 / time_taken

                    tf.compat.v2.summary.scalar(
                        'steps_per_sec', steps_per_sec, step=global_step)

                    steps_per_sec_list.append(steps_per_sec)

                    if global_step.value() - logged_step >= 100:
                        tf.logging.info(
                            'Step {} per-step time {:.3f}s loss={:.3f}'.format(
                                global_step.value(), time_taken / num_steps_per_iteration,
                                loss))
                        logged_step = global_step.value()

                    if ((int(global_step.value()) - checkpointed_step) >=
                            checkpoint_every_n):
                        manager.save()
                        checkpointed_step = int(global_step.value())


if __name__ == "__main__":

    AUTO = tf.data.experimental.AUTOTUNE
    # Original image size width=1920, height=1280
    IMAGE_SIZE = [1280, 1920]  # [640, 640] #[192, 192]
    # TPU can only use data from Google Cloud Storage
    # TPU can only load data from google cloud
    train_filenames = tf.io.gfile.glob(
        '/DATA5T/Dataset/WaymoTFRecord/train100val20/TFRecordValBig--00000-of-00005.tfrecord')
    display_dataset = load_dataset(train_filenames)
    display_dataset_iter = iter(display_dataset)
    decoded_tensors = next(display_dataset_iter)

    # decoded_tensors =next(iter(display_dataset))
    print(decoded_tensors['groundtruth_boxes'].numpy())
    print("Image width:", decoded_tensors['width'].numpy())
    print("Image height:", decoded_tensors['height'].numpy())
    print("Groundtruth classes:",
          decoded_tensors['groundtruth_classes'].numpy())

    testimage = decoded_tensors['image']
    print("Image type:", type(testimage))
    print("Image shape:", testimage.shape)
    testlabel = decoded_tensors['groundtruth_classes'].numpy()
    testboundingbox = decoded_tensors['groundtruth_boxes'].numpy()
    # show_oneimage_category(testimage, testlabel, testboundingbox, IMAGE_SIZE)
    # cv2.imwrite('result.jpg', resultimage)

    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # Start the training, ref: https://github.com/tensorflow/models/blob/master/research/object_detection/model_main_tf2.py
    pipeline_config_path = '/Developer/MyRepo/WaymoObjectDetection/2DObject/tfobjectdetection/tf_ssdresnet50_1024_pipeline_P100.config'
    model_dir = '/Developer/MyRepo/mymodels/tf_ssdresnet50_output'
    strategy = tf.compat.v2.distribute.MirroredStrategy()
    num_train_steps = 30000
    steps_per_sec_list = []
    checkpoint_every_n = 1000

    with strategy.scope():
        #in: https://github.com/tensorflow/models/blob/master/research/object_detection/model_lib_v2.py
        train_loop(
            pipeline_config_path=pipeline_config_path,
            model_dir=model_dir,
            train_steps=num_train_steps,
            use_tpu=False,
            checkpoint_every_n=1000,
            record_summaries=True)

    # with strategy.scope():
    #     #in: https://github.com/tensorflow/models/blob/master/research/object_detection/model_lib_v2.py
    #     model_lib_v2.train_loop(
    #         pipeline_config_path=pipeline_config_path,
    #         model_dir=model_dir,
    #         train_steps=num_train_steps,
    #         use_tpu=False,
    #         checkpoint_every_n=1000,
    #         record_summaries=True)
