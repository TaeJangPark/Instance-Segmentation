import os, sys, datetime, re
import re, logging, random, utils
from collections import OrderedDict
import numpy as np
import scipy.misc
import gen_cocodb as datapipe
import tensorflow as tf
import tensorflow.contrib.layers as layer
import tensorflow.contrib.slim as slim

############################################################
#  visualize Functions
############################################################
def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


############################################################
#  MobileNet Graph
############################################################

def mobilenet_graph(input_image, depth_multiplier=1, is_training=True):
    end_points = {}
    depth = lambda d: max(int(d * depth_multiplier), 8)

    def _depthwise_separable_conv(input, out_ch, stride, basename):
        scope = basename + '_depthwise'
        net = slim.separable_convolution2d(input, None, kernel_size=[3, 3],
                                           depth_multiplier=1,
                                           stride=stride,
                                           rate=1,
                                           normalizer_fn=slim.batch_norm,
                                           scope=scope)
        end_points[scope] = net

        scope = basename + '_pointwise'
        net = slim.conv2d(net, depth(out_ch), kernel_size=[1, 1],
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          scope=scope)
        end_points[scope] = net
        return net

    end_points['input'] = input_image

    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.separable_convolution2d], padding='SAME'):
            with tf.variable_scope('FeatureExtractor/MobilenetV1', [input_image], reuse=False):
                net = slim.conv2d(input_image, depth(32), kernel_size=[3, 3], stride=2, normalizer_fn=slim.batch_norm,
                                  scope='Conv2d_0')  # 1/ 2
                C1 = net
                net = _depthwise_separable_conv(net, 64, 1, 'Conv2d_1')
                net = _depthwise_separable_conv(net, 128, 2, 'Conv2d_2')  # 1 / 4
                C2 = net
                net = _depthwise_separable_conv(net, 128, 1, 'Conv2d_3')
                net = _depthwise_separable_conv(net, 256, 2, 'Conv2d_4')  # 1 / 8
                C3 = net
                net = _depthwise_separable_conv(net, 256, 1, 'Conv2d_5')
                net = _depthwise_separable_conv(net, 512, 2, 'Conv2d_6')  # 1 / 16
                C4 = net
                net = _depthwise_separable_conv(net, 512, 1, 'Conv2d_7')
                net = _depthwise_separable_conv(net, 512, 1, 'Conv2d_8')
                net = _depthwise_separable_conv(net, 512, 1, 'Conv2d_9')
                net = _depthwise_separable_conv(net, 512, 1, 'Conv2d_10')
                net = _depthwise_separable_conv(net, 512, 1, 'Conv2d_11')
                net = _depthwise_separable_conv(net, 1024, 2, 'Conv2d_12')  # 1 / 32
                net = _depthwise_separable_conv(net, 1024, 1, 'Conv2d_13')
                C5 = net

    print(C1.shape)
    print(C2.shape)
    print(C3.shape)
    print(C4.shape)
    print(C5.shape)

    return [C1, C2, C3, C4, C5]

def BatchNorm(inputs, epsilon=1e-3, suffix=''):
    """
       Assuming TxHxWxC dimensions on the tensor, will normalize over
       the H,W dimensions. Use this before the activation layer.
       This function borrows from:
           http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

       Note this is similar to batch_normalization, which normalizes each
       neuron by looking at its statistics over the batch.

       :param input_:
           input tensor of NHWC format
       """
    # Create scale + shift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    scale = tf.get_variable('scale' + suffix,
                            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('shift' + suffix,
                            initializer=tf.zeros(stat_shape[3]))

    means, vars = tf.nn.moments(inputs, axes=[1, 2],
                                          keep_dims=True)
    # Normalization
    inputs_normed = (inputs - means) / tf.sqrt(vars + epsilon)

    # Perform trainable shift.
    output = tf.add(tf.multiply(scale, inputs_normed), shift, name=suffix)
    print(output)

    return output

def Conv2D(data, out_ch, kernel, stride, padding="SAME", name=None, activation=None, is_training=True ):
    in_ch = data.get_shape().as_list()[-1]
    W = tf.get_variable(name="{}_W".format(name),
                        shape=[kernel, kernel, in_ch, out_ch], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1e-3),
                        trainable=is_training)
    feature = tf.nn.conv2d(data, W, strides=[1, stride, stride, 1], padding=padding, name=name)
    if activation is "relu":
        feature =tf.nn.relu(feature, name="{}_relu".format(name))
    return feature


############################################################
# Region Proposal Network (RPN)
############################################################
def rpn_graph(feature_map, anchors_per_location, anchor_stride, is_training=True, id=2):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the featuremap
    #       is not even.

    # Shared convolutional base of the RPN
    shared = Conv2D(feature_map, 512,
                    kernel=3,
                    stride=anchor_stride,
                    name='rpn_conv_shared_P%d' % id,
                    activation="relu",
                    is_training=is_training)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = Conv2D(shared, 2 * anchors_per_location,
                    kernel=1,
                    stride=1,
                    name='rpn_class_raw_P%d' % id,
                    padding='VALID',
                    is_training=is_training)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2], name='rpn_class_logits_P%d' % id)

    # Softmax on last dimension of BG/FG.
    rpn_probs = tf.nn.softmax(rpn_class_logits, name='rpn_class_P%d' % id)

    # Bounding box refinement. [batch, H, W, anchors per location, depth]
    # where depth is [x, y, log(w), log(h)]
    x = Conv2D(shared, anchors_per_location * 4,
                    kernel=1,
                    stride=1,
                    name='rpn_bbox_pred_P%d' % id,
                    padding='VALID',
                    is_training=is_training)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = tf.reshape(x, [tf.shape(x)[0], -1, 4], name='rpn_bbox_P%d' % id)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

############################################################
#  Proposal Layer
############################################################
def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped

def Generate_Proposal(rpn_class, rpn_bbox, proposal_count, nms_threshold, anchors, config):
    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = rpn_class[:, :, 1]
    # Box deltas [batch, num_rois, 4]
    deltas = rpn_bbox
    deltas = deltas * np.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4])
    # Base anchors
    anchors = anchors.astype(np.float32)

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.shape[0])
    ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
    scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                               config.IMAGES_PER_GPU)
    deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                               config.IMAGES_PER_GPU)
    anchors = utils.batch_slice(ix, lambda x: tf.gather(anchors, x),
                                config.IMAGES_PER_GPU,
                                names=["pre_nms_anchors"])

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = utils.batch_slice([anchors, deltas],
                              lambda x, y: apply_box_deltas_graph(x, y),
                              config.IMAGES_PER_GPU,
                              names=["refined_anchors"])

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = utils.batch_slice(boxes,
                              lambda x: clip_boxes_graph(x, window),
                              config.IMAGES_PER_GPU,
                              names=["refined_anchors_clipped"])

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Normalize dimensions to range of 0 to 1.
    normalized_boxes = boxes / np.array([[height, width, height, width]])

    # Non-max suppression
    def nms(normalized_boxes, scores):
        indices = tf.image.non_max_suppression(
            normalized_boxes, scores, proposal_count,
            nms_threshold, name="rpn_non_max_suppression")
        proposals = tf.gather(normalized_boxes, indices)
        # Pad if needed
        padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals

    proposals = utils.batch_slice([normalized_boxes, scores], nms,
                                  config.IMAGES_PER_GPU)
    batch_size = proposals.get_shape().as_list()[0]
    proposals = tf.reshape(proposals, [batch_size, proposal_count, 4], name="proposals")
    return proposals


############################################################
#  Detection Target Layer
############################################################
def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinments.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                         name="roi_assertion"),]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [anchors, crowds]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI corrdinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


def detection_target(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    # Slice the batch and run a graph for each slice
    # TODO: Rename target_bbox to target_deltas for clarity
    names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
    outputs = utils.batch_slice( [proposals, gt_class_ids, gt_boxes, gt_masks],
                                 lambda w, x, y, z: detection_targets_graph( w, x, y, z, config),
                                 config.IMAGES_PER_GPU, names=names)

    outputs[0] = tf.reshape(outputs[0], (-1, config.TRAIN_ROIS_PER_IMAGE, 4))   # rois
    outputs[2] = tf.reshape(outputs[2], (-1, config.TRAIN_ROIS_PER_IMAGE, 4))   # deltas
    outputs[3] = tf.reshape(outputs[3], (-1, config.TRAIN_ROIS_PER_IMAGE,
                                         config.MASK_SHAPE[0], config.MASK_SHAPE[1]))    # masks

    return outputs

############################################################
#  Feature Pyramid Network Heads
############################################################
def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)

def PyramidROIAlign(boxes, feature_maps, pool_shape, image_shape):
    """
    :param boxes: Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    :param feature_maps: [batch, height, width, channels] List of feature maps from different level of the feature pyramid.
    :param pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    :param image_shape: [height, width, chanells]. Shape of input image in pixels
    :return:
    """
    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
    h = y2 - y1
    w = x2 - x1
    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
    roi_level = tf.minimum(5, tf.maximum(
        2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
    roi_level = tf.squeeze(roi_level, 2)

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = tf.where(tf.equal(roi_level, level))
        level_boxes = tf.gather_nd(boxes, ix)
        # Box indicies for crop_and_resize.
        box_indices = tf.cast(ix[:, 0], tf.int32)

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        pooled.append(tf.image.crop_and_resize(
            feature_maps[i], level_boxes, box_indices, pool_shape,
            method="bilinear"))


    # Pack pooled features into one tensor
    pooled = tf.concat(pooled, axis=0)
    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = tf.concat(box_to_level, axis=0)
    box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                             axis=1)
    # Rearrange pooled features to match the order of the original boxes
    # Sort box_to_level by batch then box index
    # TF doesn't have a way to sort by two columns, so merge them and sort.
    sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
    ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
        box_to_level)[0]).indices[::-1]
    ix = tf.gather(box_to_level[:, 2], ix)
    pooled = tf.gather(pooled, ix)

    # # Re-add the batch dimension
    # pooled = tf.expand_dims(pooled, 0)
    return pooled

def fpn_classifier_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, is_training=True):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns:
        logits: [N, NUM_CLASSES] classifier logits (before softmax)
        probs: [N, NUM_CLASSES] classifier probabilities
        bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    print("======== fpn_classifier_graph ============")
    # ROI Pooling
    # Shape: [batch * num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(rois, feature_maps, [pool_size, pool_size], image_shape)
    print(x)

    # # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = layer.flatten(x)
    fc1 = layer.fully_connected(x, 1024, activation_fn=tf.nn.relu,
                                trainable=is_training,
                                biases_initializer=None,
                                scope="mrcnn_roi_fc1")
    drop1 = layer.dropout(fc1, keep_prob=0.5, is_training=is_training, scope="mrcnn_roi_drop1")
    fc2 = layer.fully_connected(drop1, 1024, activation_fn=tf.nn.relu,
                                trainable=is_training,
                                biases_initializer=None,
                                scope="mrcnn_roi_fc2")
    shared = layer.dropout(fc2, keep_prob=0.5, is_training=is_training, scope="mrcnn_roi_drop2")


    # Classifier head
    mrcnn_class_logits = layer.fully_connected(shared, num_classes, activation_fn=None,
                                               biases_initializer=None,
                                               scope='mrcnn_class_logits')


    # BBox head
    # [batch * boxes, num_classes * (dy, dx, log(dh), log(dw))]
    mrcnn_bbox = layer.fully_connected(shared, num_classes * 4, activation_fn=None,
                              biases_initializer=None,
                              scope='mrcnn_bbox_fc')


    # # Two 1024 FC layers (implemented with Conv2D for consistency)
    # x = Conv2D(x, 1024, pool_size, stride=1, padding='VALID', name="mrcnn_class_conv1")
    # x = layer.batch_norm(x,trainable=is_training, activation_fn=tf.nn.relu, scope="mrcnn_class_bn1")
    # print(x)
    # x = Conv2D(x, 1024, 1, stride=1, name="mrcnn_class_conv2")
    # x = layer.batch_norm(x,trainable=is_training, activation_fn=tf.nn.relu, scope="mrcnn_class_bn2")
    # print(x)
    #
    # shared = layer.flatten(x)
    # print(shared)

    return mrcnn_class_logits, mrcnn_bbox
    # return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes, is_training=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    print("======== build_fpn_mask_graph ============")
    # ROI Pooling
    # Shape: [batch * num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(rois, feature_maps, [pool_size, pool_size], image_shape)

    # Conv layers
    for i in range(4):
        x = Conv2D(x, 256, kernel=3, stride=1, padding='SAME', name="mrcnn_mask_conv%d"%(i+1))
        x = layer.batch_norm(x, trainable=is_training, activation_fn=tf.nn.relu, scope="mrcnn_mask_bn%d"%(i+1))

    x = tf.image.resize_images(x, [28, 28])
    x = Conv2D(x, 256, kernel=3, stride=1, activation="relu", padding='SAME', name="mrcnn_mask_upconv")
    x = Conv2D(x, num_classes, kernel=1, stride=1, padding='SAME', name="mrcnn_mask_conv")
    x = tf.nn.sigmoid(x, name='mrcnn_mask')
    # shape = [-1, num_rois_per_image] + x.get_shape().as_list()[1:]
    # x = tf.reshape(x, shape)

    return x

############################################################
#  Loss Functions
############################################################
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    # Smooth-L1 Loss
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), dtype=tf.float32)
    loss = (less_than_one * 0.5 * diff **2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    # Crossentropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_class,
                                                          logits=rpn_class_logits)
    loss = tf.cond(tf.size(loss)>0, lambda : tf.reduce_mean(loss), lambda : tf.constant(0.0))

    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    # # TODO: use smooth_l1_loss() rather than reimplementing here
    # diff = tf.abs(target_bbox - rpn_bbox)
    # less_than_one = tf.cast(tf.less(diff, 1.0), dtype=tf.float32)
    # loss = (less_than_one * 0.5 * diff **2) + (1 - less_than_one) * (diff - 0.5)
    #
    # loss = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))

    loss = tf.cond(tf.size(target_bbox) > 0, lambda: smooth_l1_loss(y_true=target_bbox, y_pred=rpn_bbox),
                   lambda: tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    print('loss', loss)

    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    print("========== mrcnn_class_loss_graph ================")
    target_class_ids = tf.cast(target_class_ids, tf.int64)

    # Find predictions of classes that are not in the dataset.
    print(target_class_ids)
    print(pred_class_logits)
    print(active_class_ids)
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    print(pred_class_ids)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    print(pred_active)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.

    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, pred_bbox.get_shape().as_list()[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    loss = tf.cond(tf.size(target_bbox) > 0, lambda: smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                   lambda: tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    # loss = tf.reshape(loss, [1, 1])

    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    print("========== mrcnn_mask_loss_graph ================")
    print(target_masks)
    print(target_class_ids)
    print(pred_masks)

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = tf.cond(tf.size(y_true) > 0,
                   lambda: tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred),
                   lambda: tf.constant(0.0))

    loss = tf.reduce_mean(loss)
    # loss = tf.reshape(loss, [1, 1])
    print(loss)
    return loss


############################################################
#  Detection Layer for Inference
############################################################
def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes

def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    print(rois.shape, probs.shape, deltas.shape, window.shape)
    # Class IDs per ROI
    class_ids = np.argmax(probs, axis=1)
    # Class probability of the top class of each ROI
    class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = utils.apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(
            keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        class_ids[keep][..., np.newaxis],
                        class_scores[keep][..., np.newaxis]))
    return result

def DetectionLayer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
        print(rois.shape, mrcnn_class.shape, mrcnn_bbox.shape, image_meta.shape)

        detections_batch = []
        _, _, window, _ = datapipe.parse_image_meta(image_meta)
        for b in range(config.BATCH_SIZE):
            detections = refine_detections(
                rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], config)
            # Pad with zeros if detections < DETECTION_MAX_INSTANCES
            gap = config.DETECTION_MAX_INSTANCES - detections.shape[0]
            assert gap >= 0
            if gap > 0:
                detections = np.pad(
                    detections, [(0, gap), (0, 0)], 'constant', constant_values=0)
            detections_batch.append(detections)

        # Stack detections and cast to float32
        # TODO: track where float64 is introduced
        detections_batch = np.array(detections_batch).astype(np.float32)
        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
        return np.reshape(detections_batch, [config.BATCH_SIZE, config.DETECTION_MAX_INSTANCES, 6])

    # Return wrapped function
    return tf.py_func(wrapper, [rois, mrcnn_class, mrcnn_bbox, image_meta], tf.float32)

############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir, anchors):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.anchors = anchors
        self.set_log_dir()
        self.outputs={}
        self.networks = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """
        Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']
        if mode == 'training':
            self.is_training = True
        else:
            self.is_training = False

        print("is_training", self.is_training)

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        self.input_image = tf.placeholder(dtype=tf.float32,
            shape=[None]+config.IMAGE_SHAPE.tolist(), name="input_image")

        self.input_image_meta = tf.placeholder(dtype=tf.float32, shape=[None, None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            self.input_rpn_match = tf.placeholder(dtype=tf.int32, shape=[None, None, 1], name="input_rpn_match")
            self.input_rpn_bbox = tf.placeholder(dtype=tf.float32, shape=[None, None, 4], name="input_rpn_bbox")

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            self.input_gt_class_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_gt_class_ids")

            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            self.input_gt_boxes = tf.placeholder(dtype=tf.float32, shape=[None, None, 4], name="input_gt_boxes")
            # Normalize coordinates
            h, w = self.input_image.get_shape().as_list()[1:3]
            image_scale = tf.cast(tf.stack([h, w, h, w], axis=0), tf.float32)
            gt_boxes = self.input_gt_boxes/ image_scale

            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                self.input_gt_masks = tf.placeholder(dtype=bool,
                    shape=[None, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks")
            else:
                self.input_gt_masks = tf.placeholder(dtype=bool,
                    shape=[None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = mobilenet_graph(self.input_image, depth_multiplier=1.0, is_training=self.is_training)
        # Top-down Layers

        # TODO: add assert to varify feature map sizes match what's in config
        with tf.variable_scope('FPN', regularizer=layer.l2_regularizer(self.config.WEIGHT_DECAY)):
            # Pyramid 5
            convP5 = Conv2D(C5, 256, 1, 1, name="fpn_C5P5", activation=None, is_training=self.is_training)

            # Pyramid 4
            target_shape = tf.shape(C4)
            up_p5 = tf.image.resize_bilinear(convP5, [target_shape[1], target_shape[2]], name='fpn_p5upsampled')
            # 1x1 conv
            conv_C4 = Conv2D(C4, 256, 1, 1, name="fpn_C4P4", activation=None, is_training=self.is_training)
            # add P5 and C4
            fusion_P4 =tf.add(up_p5, conv_C4, name='fpn_P5C4_add')

            # Pyramid 3
            target_shape = tf.shape(C3)
            up_p4 = tf.image.resize_bilinear(fusion_P4, [target_shape[1], target_shape[2]], name='fpn_p4upsampled')
            # 1x1 conv
            conv_C3 = Conv2D(C3, 256, 1, 1, name="fpn_C3P3", activation=None, is_training=self.is_training)
            # add P4 and C3
            fusion_P3 =tf.add(up_p4, conv_C3, name='fpn_P4C3_add')

            # Pyramid 2
            target_shape = tf.shape(C2)
            up_p3 = tf.image.resize_bilinear(fusion_P3, [target_shape[1], target_shape[2]], name='fpn_p5upsampled')

            # 1x1 conv
            conv_C2 = Conv2D(C2, 256, 1, 1, name="fpn_C2P2", activation=None, is_training=self.is_training)
            # add P3 and C2
            fusion_P2 =tf.add(up_p3, conv_C2, name='fpn_P3C2_add')

            # P5 3x3 conv
            P5 = Conv2D(convP5, 256, 3, 1, name="fpn_P5", activation=None, is_training=self.is_training)
            # P4 3x3 conv
            P4 = Conv2D(fusion_P4, 256, 3, 1, name="fpn_P4", activation=None, is_training=self.is_training)
            # P3 3x3 conv
            P3 = Conv2D(fusion_P3, 256, 3, 1, name="fpn_P3", activation=None, is_training=self.is_training)
            # P2 3x3 conv
            P2 = Conv2D(fusion_P2, 256, 3, 1, name="fpn_P2", activation=None, is_training=self.is_training)
            # P6
            P6 = tf.nn.max_pool(P5, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME', name="fpn_P6" )

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # RPN Model
        with tf.variable_scope('RPN', regularizer=layer.l2_regularizer(self.config.WEIGHT_DECAY)):
            layer_outputs = []
            anchors_per_location = len(config.RPN_ANCHOR_RATIOS)
            id = 2

            # Loop through pyramid layers
            for p in rpn_feature_maps:
                outputs = rpn_graph(p, anchors_per_location, config.RPN_ANCHOR_STRIDE, self.is_training, id=id)
                id += 1
                layer_outputs.append(outputs)

            # Concatenate layer outputs
            # Convert from list of lists of level outputs to list of lists
            # of outputs across levels.
            # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
            output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
            outputs = list(zip(*layer_outputs))

            outputs = [tf.concat(values=list(o), axis=1, name=n)
                       for o, n in zip(outputs, output_names)]

            rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = 0
        if self.is_training:
            proposal_count = config.POST_NMS_ROIS_TRAINING
        else:
            proposal_count = config.POST_NMS_ROIS_INFERENCE

        print(proposal_count, rpn_class)
        rpn_rois = Generate_Proposal(rpn_class, rpn_bbox,
                                      proposal_count=proposal_count,
                                      nms_threshold=config.RPN_NMS_THRESHOLD,
                                      anchors=self.anchors,
                                      config=config)
        print("rpn_rois", rpn_rois)

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_class_ids = datapipe.parse_image_meta_graph(self.input_image_meta)

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask = detection_target(rpn_rois,
                                                                                self.input_gt_class_ids,
                                                                                gt_boxes, self.input_gt_masks,
                                                                                self.config)

            print('rois', rois, target_class_ids, target_bbox, target_mask)

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            with tf.variable_scope('mrcnn', regularizer=layer.l2_regularizer(self.config.WEIGHT_DECAY)):
                mrcnn_class_logits, mrcnn_bbox = fpn_classifier_graph(rois, mrcnn_feature_maps,
                                                                                   config.IMAGE_SHAPE,
                                                                                   config.POOL_SIZE,
                                                                                   config.NUM_CLASSES,
                                                                                   is_training=self.is_training)
                # Reshape to [batch, boxes, num_classes]
                mrcnn_class_logits = tf.reshape(mrcnn_class_logits, (-1, config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES))
                mrcnn_class = tf.nn.softmax(mrcnn_class_logits)
                print(mrcnn_class)

                # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
                mrcnn_bbox = tf.reshape(mrcnn_bbox, [-1, config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4], name="mrcnn_bbox")


                mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                  config.IMAGE_SHAPE,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  is_training=self.is_training)

                shape = [-1, config.TRAIN_ROIS_PER_IMAGE] + mrcnn_mask.get_shape().as_list()[1:]
                mrcnn_mask = tf.reshape(mrcnn_mask, shape)
                print(mrcnn_mask, target_mask)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = tf.identity(rois, name="output_rois")
            print(output_rois)

            # Losses
            rpn_class_loss = rpn_class_loss_graph(self.input_rpn_match, rpn_class_logits)
            rpn_bbox_loss = rpn_bbox_loss_graph(self.config, self.input_rpn_bbox, self.input_rpn_match, rpn_bbox)
            class_loss = mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids)
            bbox_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox)
            mask_loss = mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)
            tf.add_to_collection(tf.GraphKeys.LOSSES, rpn_class_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, rpn_bbox_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, class_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, bbox_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, mask_loss)

        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            with tf.variable_scope('mrcnn', regularizer=layer.l2_regularizer(self.config.WEIGHT_DECAY)):
                num_rois = rpn_rois.get_shape().as_list()[1]
                print(num_rois)
                mrcnn_class_logits, mrcnn_bbox = fpn_classifier_graph(rpn_rois, mrcnn_feature_maps,
                                                                      config.IMAGE_SHAPE,
                                                                      config.POOL_SIZE,
                                                                      config.NUM_CLASSES,
                                                                      is_training=self.is_training)

                batch_size = np.asarray(rpn_rois.get_shape().as_list()[0])
                mrcnn_class_logits = tf.reshape(mrcnn_class_logits, [batch_size, num_rois, config.NUM_CLASSES])
                mrcnn_class = tf.nn.softmax(mrcnn_class_logits)

                mrcnn_bbox = tf.reshape(mrcnn_bbox, [batch_size, num_rois, config.NUM_CLASSES, 4], name="mrcnn_bbox")

                print("fpn_classifier_graph", rpn_rois, mrcnn_class_logits, mrcnn_bbox)


                # Detections
                # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
                detections = DetectionLayer(config, rpn_rois, mrcnn_class, mrcnn_bbox, self.input_image_meta)
                print("detections", detections)

                # Convert boxes to normalized coordinates
                # TODO: let DetectionLayer return normalized coordinates to avoid
                #       unnecessary conversions
                h, w = config.IMAGE_SHAPE[:2]
                detection_boxes = detections[..., :4] / np.array([h, w, h, w])

                # Create masks for detections
                mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                  config.IMAGE_SHAPE,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  is_training=self.is_training)

                shape = [-1, config.DETECTION_MAX_INSTANCES] + mrcnn_mask.get_shape().as_list()[1:]
                mrcnn_mask = tf.reshape(mrcnn_mask, shape)
                print("mrcnn_mask", mrcnn_mask)

                self.outputs['detections'] = detections
                self.outputs['mrcnn_class'] = mrcnn_class
                self.outputs['mrcnn_bbox'] = mrcnn_bbox
                self.outputs['mrcnn_mask'] = mrcnn_mask
                self.outputs['rois'] = rpn_rois
                self.outputs['rpn_class'] = rpn_class
                self.outputs['rpn_bbox'] = rpn_bbox

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.ckpt".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)
