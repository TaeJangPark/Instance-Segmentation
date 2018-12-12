import os, random
import numpy as np
import scipy.misc as sm
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import libs.nets.model as modellib
from libs.configs import config
import utils, gen_cocodb, visualize

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


cat_list =['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
           'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def print_tensors_in_checkpoint_file(file_name, all_tensors):
    """
    Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
      in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
        file_name: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
    """
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            print("tensor_name: ", key)
            # print(reader.get_tensor(key))


class InferenceConfig(config.Config):
    # Give the configuration a recognizable name
    NAME = "coco"

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def inference():
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs
    LOG_DIR = os.path.join(ROOT_DIR, "output/logs")
    MODEL_DIR = os.path.join(ROOT_DIR, "output/training")
    dataset_path = os.path.join(ROOT_DIR, 'data/coco')

    config = InferenceConfig()
    config.display()

    dataset_val = gen_cocodb.CocoDataSet()
    dataset_val.load_coco(dataset_path, "minival", year="2014", auto_download=False)
    dataset_val.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))

    image_id = random.choice(dataset_val.image_ids)
    # image, image_meta, gt_class_id, gt_bbox, gt_mask = dataset_val.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)
    # info = dataset_val.image_info[image_id]
    # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
    #                                        dataset_val.image_reference(image_id)))

    image = dataset_val.load_image(image_id)
    images = np.expand_dims(image, axis=0)
    molded_images, image_metas, windows = gen_cocodb.mold_inputs(images, config)
    print(molded_images.shape, image_metas.shape, windows.shape)

    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)
    with tf.device('/device:CPU:0'):
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=LOG_DIR, anchors=anchors)
    print(len(model.outputs))

    feed_dict = {model.input_image: molded_images, model.input_image_meta: image_metas}

    detections = model.outputs['detections']
    mrcnn_class = model.outputs['mrcnn_class']
    mrcnn_bbox = model.outputs['mrcnn_bbox']
    mrcnn_mask = model.outputs['mrcnn_mask']

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.device('/device:CPU:0'):
        with tf.Session() as sess:
            sess.run(init_op)
            # saver.restore(sess, "output/training/mrcnn.ckpt-96000")
            ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
            """ resotre checkpoint of Backbone network """
            if ckpt is not None:
                ckpt_path = tf.train.latest_checkpoint(MODEL_DIR)
                # ckpt_path = FLAGS.checkpoint_model
                saver.restore(sess, ckpt_path)
            else:
                ckpt_path = "output/training/mrcnn.ckpt-96000"
                saver.restore(sess, ckpt_path)
            print('ckpt_path', ckpt_path)

            pre_nms_anchors = sess.graph.get_tensor_by_name("pre_nms_anchors:0")
            refined_anchors = sess.graph.get_tensor_by_name("refined_anchors:0")
            refined_anchors_clipped = sess.graph.get_tensor_by_name("refined_anchors_clipped:0")
            print(pre_nms_anchors)
            print(refined_anchors)
            print(refined_anchors_clipped)

            detect, pred_class, pred_bbox, pred_mask = sess.run([detections, mrcnn_class, mrcnn_bbox, mrcnn_mask],
                                                                feed_dict=feed_dict)

            print(detect.shape, pred_class.shape, pred_bbox.shape, pred_mask.shape)

            # Process detections
            final_rois, final_class_ids, final_scores, final_masks = gen_cocodb.unmold_detections(detect[0],
                                                                                                  pred_mask[0],
                                                                                                  image.shape,
                                                                                                  windows[0])

            ax = get_ax(1)
            visualize.display_instances(image, final_rois, final_masks, final_class_ids,
                                        dataset_val.class_names, final_scores, ax=ax,
                                        title="Predictions")
            print(final_rois.shape, final_class_ids.shape, final_scores.shape, final_masks.shape)
            print(final_class_ids)
            print(final_scores)
            print(final_rois)


if __name__ == '__main__':
    inference()
