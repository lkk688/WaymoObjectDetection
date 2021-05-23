import wod_latency_submission

from glob import glob
import time
import os
from pathlib import Path
import numpy as np

import visualization_util

from object_detection.utils import label_map_util

if __name__ == "__main__":
    # test the above functions: convert a Frame proto into a dictionary
    # convert_frame_to_dict
    # "/data/cmpe249-f20/WaymoKittitMulti/dict_train0"
    base_dir = "/Developer/3DObject"
    base_dir = Path(base_dir)
    filename = "1_step10_10017090168044687777_6380_000_6400_000.npz"

    Final_array = np.load(base_dir / filename,
                          allow_pickle=True, mmap_mode='r')
    data_array = Final_array['arr_0']
    array_len = len(data_array)
    # 20, 200 frames in one file, downsample by 10
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))  # numpy.ndarray

    # for frameid in range(array_len):
    frameid = 5
    print("frameid:", frameid)
    # {'key':key, 'context_name':context_name, 'framedict':framedict}
    convertedframesdict = data_array[frameid]
    frame_timestamp_micros = convertedframesdict['key']
    context_name = convertedframesdict['context_name']
    framedict = convertedframesdict['framedict']
    # 10017090168044687777_6380_000_6400_000
    print('context_name:', context_name)
    print('frame_timestamp_micros:', frame_timestamp_micros)  # 1550083467346370

    wod_latency_submission.initialize_model()

    required_field = wod_latency_submission.DATA_FIELDS
    print(required_field)

    #result = wod_latency_submission.run_model(framedict[required_field[0]], framedict[required_field[1]])
    #result = wod_latency_submission.run_model(**framedict)
    Front_image = framedict[required_field[0]]
    result = wod_latency_submission.run_model(Front_image)
    print(result)

    output_path = "./test.png"
    visualization_util.save_image_array_as_png(Front_image, output_path)

    image_np_with_detections = Front_image.copy()
    label_map_path = '2DObject/tfobjectdetection/waymo_labelmap.txt'
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    visualization_util.visualize_boxes_and_labels_on_image_array(image_np_with_detections, result['boxes'], result['classes'], result['scores'], category_index, use_normalized_coordinates=False,
                                                             max_boxes_to_draw=200,
                                                             min_score_thresh=.08,
                                                             agnostic_mode=False)
    visualization_util.save_image_array_as_png(
        image_np_with_detections, "./testresult.png")
# plt.figure(figsize=(12,16))
# plt.imshow(image_np_with_detections)
# plt.show()
