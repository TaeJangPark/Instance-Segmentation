import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

import libs.configs.config as config
import libs.nets.model as modellib
from libs.deployment import model_deploy
from PIL import Image, ImageDraw
import scipy.misc as sm

import gen_cocodb
import utils




FLAGS = tf.app.flags.FLAGS

cls_name = np.array([  'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush'])

def mask_checking(image, gt_boxes, gt_masks, norm = False):
    img = np.squeeze(image)

    if norm:
        img = ((img + 1.0) / 2.0) * 255
        img = img.astype(np.uint8)

    print(img.shape)

    colors = ["blue", "black", "brown", "red", "yellow", "green", "orange", "beige", "turquoise", "pink"]

    j = Image.fromarray(img)
    draw = ImageDraw.Draw(j)
    a = np.zeros((image.shape[1], image.shape[2]), np.uint8)
    dst = np.zeros((image.shape[1], image.shape[2], 4))

    for i in range(min(len(gt_boxes), 10)):
        bbox = gt_boxes[i][:4]
        m = gt_masks[i]
        a[m > 0] = m[m > 0] * 255
        draw.rectangle(bbox, outline=colors[i])
        print('class', gt_boxes[i][4])

    dst[:, :, 0] = img[..., 0]
    dst[:, :, 1] = img[..., 1]
    dst[:, :, 2] = img[..., 2]
    dst[:, :, 3] = a
    sm.imsave('images/mask.png', dst)
    j.save('images/bbox.png')


def net_checking(net):
    im = net['input'][0]
    rgb = ((im[:, :, ::-1] + 1.0)/2.0) * 255.0
    C1 = net['conv1'][0]
    C2 = net['block1/unit_2/bottleneck'][0]
    C3 = net['block2/unit_3/bottleneck'][0]
    C4 = net['block3/unit_5/bottleneck'][0]
    C5 = net['block4/unit_3/bottleneck'][0]

    print(C1.shape, C2.shape, C3.shape, C4.shape, C5.shape)

    """ save conv1 image from 0 channel to 9 channel """
    for i in range(10):
        sm.imsave('images/C1_%d.jpg' % i, C1[:, :, i * 5])
        sm.imsave('images/C2_%d.jpg' % i, C2[:, :, i * 5])
        sm.imsave('images/C3_%d.jpg' % i, C3[:, :, i * 5])
        sm.imsave('images/C4_%d.jpg' % i, C4[:, :, i * 5])
        sm.imsave('images/C5_%d.jpg' % i, C5[:, :, i * 5])
def pyramid_checking(pyramid):
    for i in pyramid:
        print(i, pyramid[i].shape)
        p = pyramid[i][0]

        """ save conv1 image from 0 channel to 9 channel """
        for j in range(10):
            sm.imsave('images/%s_%d.jpg' % (i, j), p[:, :, j * 5])


def anchor_checking(image, RPNs):
    colors = ["blue", "black", "brown", "red", "yellow", "green", "orange", "beige", "turquoise", "pink"]

    for i in range(5, 1, -1):
        box = RPNs['rpn']['P%d' % i]['box'][0]
        cls = RPNs['rpn']['P%d' % i]['cls'][0]
        anchor = RPNs['rpn']['P%d' % i]['anchor']
        print(i, box.shape, cls.shape, anchor.shape)

        dst = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8)
        dst[:, :, 0] = ((image[0][..., 2] + 1) / 2) * 255
        dst[:, :, 1] = ((image[0][..., 1] + 1) / 2) * 255
        dst[:, :, 2] = ((image[0][..., 0] + 1) / 2) * 255

        img = Image.fromarray(dst)
        draw = ImageDraw.Draw(img)

        cy = anchor.shape[0] / 2 - 1
        cx = anchor.shape[1] / 2 - 1

        for n in range(9):
            a_box = anchor[cy, cx, 4 * n:4 * (n + 1)]

            draw.rectangle(a_box, outline=colors[n])

        img.save('images/P%d_anchors.jpg' % i)

def checking_RoIs(pyramid, ori_image, RoI):
    assigned_rois = RoI['roi']['box']
    assigned_batch_inds = RoI['roi']['batch']
    crop_box = RoI['roi']['cropped_rois']
    print(crop_box.shape, crop_box[0].shape)

    cnt = 0
    for i in range(len(assigned_rois)):
        print(i, assigned_rois[i].shape, assigned_batch_inds[i].shape)
        image = pyramid['P%d' % (i + 2)]
        re_img = sm.imresize(ori_image[0], (image.shape[1], image.shape[2]))

        dst = np.zeros((re_img.shape[0], re_img.shape[1], 3), dtype=np.uint8)
        dst[:, :, 0] = re_img[..., 2]
        dst[:, :, 1] = re_img[..., 1]
        dst[:, :, 2] = re_img[..., 0]

        img = Image.fromarray(dst)
        draw = ImageDraw.Draw(img)

        bbox = assigned_rois[i]
        boxes = bbox / (2 ** (i + 2) + 0.0)

        print(image.shape[1], image.shape[2])
        lenth = len(bbox)
        if lenth is not 0:
            for j in range(1):
                colors = np.random.rand(1, 3) * 255
                c = []
                for t in range(3):
                    c.append(int(colors[0][t]))
                c = tuple(c)
                draw.rectangle(boxes[j], outline=c)

            img.save('images/assigned__crop_%d.jpg' % i)
        sm.imsave('images/cropped_P%d.jpg' % (i), crop_box[cnt, :, :, 0])
        cnt = cnt + lenth - 1




def checking_gt_box_pyramid(ori_image, pyramid, gt_boxes):
    ori_img = np.squeeze(ori_image)

    ori_img = ((ori_img + 1.0) / 2.0) * 255
    ori_img = ori_img.astype(np.uint8)

    colors = []
    j = Image.fromarray(ori_img)
    draw = ImageDraw.Draw(j)

    for i in range(len(gt_boxes)):
        bbox = gt_boxes[i][:4]
        rand = np.random.rand(1, 3) * 255
        c = []
        for t in range(3):
            c.append(int(rand[0][t]))
        colors.append(tuple(c))
        draw.rectangle(bbox, outline=colors[i])
    j.save('images/gt_bbox.png')

    k0 = 4
    min_k = 2
    max_k = 5
    if gt_boxes.size > 0:
        ws = gt_boxes[:, 2] - gt_boxes[:, 0]
        hs = gt_boxes[:, 3] - gt_boxes[:, 1]
        areas = ws * hs
        k = np.floor(k0 + np.log2(np.sqrt(areas) / 224))
        inds = np.where(k < min_k)[0]
        k[inds] = min_k
        inds = np.where(k > max_k)[0]
        k[inds] = max_k

    assigned_layers = np.reshape(k, [-1])

    print(gt_boxes.shape)
    assigned_tensors = []
    for t in [gt_boxes]:
        split_tensors = []
        for l in [2, 3, 4, 5]:
            inds = np.where(np.equal(assigned_layers, l))
            inds = np.reshape(inds, [-1])
            split_tensors.append(t[inds])
        assigned_tensors.append(split_tensors)

    for l in range(5, 1, -1):
        gt_bbox = assigned_tensors[0][l - 2]
        print(len(gt_bbox))
        image = pyramid['P%d' % (l)]
        re_img = sm.imresize(ori_image[0], (image.shape[1], image.shape[2]))
        dst1 = np.zeros((re_img.shape[0], re_img.shape[1], 3), dtype=np.uint8)
        dst1[:, :, 0] = re_img[..., 2]
        dst1[:, :, 1] = re_img[..., 1]
        dst1[:, :, 2] = re_img[..., 0]

        img1 = Image.fromarray(dst1)
        draw1 = ImageDraw.Draw(img1)
        for i in range(len(gt_bbox)):
            bbox = gt_bbox[i][0:4]
            boxes = bbox / (2 ** l)
            draw1.rectangle(boxes, outline=colors[i])
        img1.save('images/gt_bbox_P%d.jpg' % l)


def _get_learning_rate(num_sample_per_epoch, global_step):
    decay_step = int((num_sample_per_epoch / cfg.FLAGS.batch_size) * cfg.FLAGS.num_epochs_per_decay)
    return tf.train.exponential_decay(cfg.FLAGS.learning_rate,
                                      global_step,
                                      decay_step,
                                      cfg.FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')



def _get_restore_vars(scope):
    print("======== restore_variables ==============")
    print(scope)
    mobilenet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    variables_to_restore = []

    # exclusions = ['mask/Conv', 'mask/Conv_1/', 'mask/Conv_2/', 'mask/Conv_3/', 'mask/Conv_4/']
    # variables_to_restore = []
    for var in mobilenet_vars:
        s = var.op.name
        if s.find("Momentum") == -1:
            variables_to_restore.append(var)
    print('final')
    for i in variables_to_restore:
        print(i)
    return variables_to_restore

def set_trainable(train_layers):
    # Pre-defined layer regular expressions
    print("======== set_trainable ==============")

    if train_layers == "heads":
       exclusions = ['FeatureExtractor/MobilenetV1']
    elif train_layers == "4+":
       exclusions = ['FeatureExtractor/MobilenetV1/Conv2d_0/',
                     'FeatureExtractor/MobilenetV1/Conv2d_1_depthwise', 'FeatureExtractor/MobilenetV1/Conv2d_1_pointwise',
                     'FeatureExtractor/MobilenetV1/Conv2d_2_depthwise', 'FeatureExtractor/MobilenetV1/Conv2d_2_pointwise',
                     'FeatureExtractor/MobilenetV1/Conv2d_3_depthwise', 'FeatureExtractor/MobilenetV1/Conv2d_3_pointwise',
                     'FeatureExtractor/MobilenetV1/Conv2d_4_depthwise', 'FeatureExtractor/MobilenetV1/Conv2d_4_pointwise']
    else:
        return tf.global_variables()

    variables_to_train = []
    for var in tf.global_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_train.append(var)

    for i in variables_to_train:
        print(i)
    print("======================")
    return variables_to_train


def print_tensors_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        print("tensor_name: ", key)


class CocoConfig(config.Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def data_generator(dataset, config, anchors, shuffle=True ):
    """
    A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
                 horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                     network classifier and mask heads. Useful if training
                     the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
            deltas, and masks). Typically for debugging or visualizations because
            in trainig detection targets are generated by DetectionTargetLayer.
    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
        - images: [batch, H, W, C]
        - image_meta: [batch, size of image meta]
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image unless use_mini_mask is True, in which
                    case they are defined in MINI_MASK_SHAPE.
    outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bbox deltas,
            and masks.
    """
    b = 0  # batch index
    num_epoch = 0
    batch_size = config.BATCH_SIZE
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    while True:
        try:
            image_index = (image_index + 1) % len(image_ids)
            # shuffle images if at the start of an epoch.
            if image_index == 0:
                np.random.shuffle(image_ids)
                num_epoch += 1

            # Get gt_boxes and gt_masks for image.
            image_id = image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                coco_train.load_image_gt(coco_train,
                                         config, image_id,
                                         augment=True,
                                         use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = utils.build_rpn_targets(image.shape, anchors, gt_class_ids, gt_boxes, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                               config.MAX_GT_INSTANCES))
                else:
                    batch_gt_masks = np.zeros((batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                print("Gt is too much!!")
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_images[b] = gen_cocodb.mold_image(image.astype(np.float32), config)
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs =[batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                         batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, num_epoch]
                yield inputs

                b = 0  # batch item index
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            print("Error processing image {}".format(dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def train(train_dataset, config, lr, train_layers, epochs):

    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(num_clones=config.GPU_COUNT,
                                                      clone_on_cpu=False,
                                                      replica_id=0,
                                                      num_replicas=1,
                                                      num_ps_tasks=0)

        with tf.device(deploy_config.variables_device()):
            print(deploy_config.variables_device())
            global_step = tf.train.create_global_step()

        with tf.device(deploy_config.inputs_device()):
            print(deploy_config.inputs_device())
            with tf.name_scope('coco_data_generator'):
                train_generator = data_generator(train_dataset, config, anchors, shuffle=True)

        models =[]
        def clone_fn():
            model = modellib.MaskRCNN(mode=mode, config=config, model_dir=DEFAULT_LOGS_DIR, anchors=anchors)
            models.append(model)

            losses = tf.get_collection(tf.GraphKeys.LOSSES)
            model_loss = tf.add_n(losses)
            return model_loss


        clones = model_deploy.create_clones(deploy_config, clone_fn)
        first_clone_scope = deploy_config.clone_scope(0)
        print(first_clone_scope)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for loss in tf.get_collection(tf.GraphKeys, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        #########################################
        # Configure the optimization procedure. #
        #########################################
        print(deploy_config.optimizer_device())
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=config.LEARNING_MOMENTUM, name='Momentum')
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        variables_to_train = set_trainable(train_layers)

        total_loss, clones_gradients = model_deploy.optimize_clones(clones,
                                                                    optimizer,
                                                                    var_list=variables_to_train)
        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)

        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))
        print(total_loss)

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        summary_writer = tf.summary.FileWriter(models[0].log_dir, graph=tf.Session().graph)

        """ set saver for saving final model and backbone model for restore """
        # variables_to_restore = _get_restore_vars('FeatureExtractor/MobilenetV1')
        # re_saver = tf.train.Saver(var_list=variables_to_restore)

        saver = tf.train.Saver(max_to_keep=3)
        """ Set Gpu Env """
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        """ Starting Training..... """
        gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_opt)) as sess:
            sess.run(init_op)
            # re_saver.restore(sess, 'data/pretrained_models/mobilenet_v1_coco/model.ckpt')
            ckpt = tf.train.get_checkpoint_state("output/training")
            """ resotre checkpoint of Backbone network """
            if ckpt:
                lastest_ckpt = tf.train.latest_checkpoint("output/training")
                print('lastest', lastest_ckpt)
                saver.restore(sess, lastest_ckpt)
            try:
                while True:
                    feed_dict={learning_rate:lr}
                    inputs = train_generator.next()
                    num_epoch = inputs[7]
                    for i in range(len(clones)):
                        s = 2*i
                        e = 2*(i+1)
                        feed_dict[models[i].input_image] = inputs[0][s:e, :]
                        feed_dict[models[i].input_image_meta] = inputs[1][s:e, :]
                        feed_dict[models[i].input_rpn_match] = inputs[2][s:e, :]
                        feed_dict[models[i].input_rpn_bbox] = inputs[3][s:e, :]
                        feed_dict[models[i].input_gt_class_ids] = inputs[4][s:e, :]
                        feed_dict[models[i].input_gt_boxes] = inputs[5][s:e, :]
                        feed_dict[models[i].input_gt_masks] = inputs[6][s:e, :]

                    _, loss, current_step, summary =  sess.run([update_op, total_loss,
                                                                global_step, summary_op],
                                                               feed_dict=feed_dict)
                    print ("""iter %d : total-loss %.4f """ %(current_step, loss))

                    if np.isnan(loss) or np.isinf(loss):
                        print('isnan or isinf', loss)
                        raise
                    if current_step % 1000 == 0:
                        # write summary
                        # summary = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary, current_step)
                        summary_writer.flush()

                    if current_step % 3000 == 0:
                        # Save a checkpoint
                        save_path = 'output/training/mrcnn.ckpt'
                        saver.save(sess, save_path, global_step=current_step)

                    if num_epoch > epochs:
                        print("num epoch : %d and training End!!!" % num_epoch)
                        break
            except Exception as ex:
                print('Error occured!!!! => ', ex)
            finally:
                print("Final!!")
                saver.save(sess, 'output/models/mrcnn_final.ckpt', write_meta_graph=False)



if __name__ == "__main__":
    mode = "training"
    print("mode ==> %%s!!", mode)

    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "output/logs")

    dataset_path = os.path.join(ROOT_DIR, 'data/coco')
    config = CocoConfig()

    print('dataset_path: ', dataset_path)
    coco_train = gen_cocodb.CocoDataSet()
    coco_train.load_coco(dataset_path, "train", year="2014", auto_download=False)
    coco_train.load_coco(dataset_path, "valminusminival", year="2014", auto_download=False)
    coco_train.prepare()

    # coco_val = gen_cocodb.CocoDataSet()
    # coco_val.load_coco(dataset_path, "minival", year="2014", auto_download=False)
    # coco_val.prepare()

    # generate base anchors with from [256, 256] to [16, 16].




    # iter : 2444360
    train(coco_train, config, lr=config.LEARNING_RATE, train_layers="heads", epochs=40)

    # iter : 7333080
    # train(coco_train, model, config, lr=config.LEARNING_RATE, train_layers="4+", epochs=120)

    # iter : 9777440
    # train(coco_train, model, config, lr=config.LEARNING_RATE / 10, train_layers="all", epochs=160)