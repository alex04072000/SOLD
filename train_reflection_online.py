from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from model import Decomposition_Net_Translation
from model import ImageReconstruction_reflection_arbitraryFrameNum_large_FBconcat_AvgMeanPool as ImageReconstruction_reflection_arbitraryFrameNum
from warp_utils import dense_image_warp
import cv2
import glob

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './temp_online_ckpt/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 210,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_string('training_scene', None,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('height', 192,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('width', 320,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('GPU_ID', '0',
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('training_data_path', 'reflection_imgs/', """Number of batches to run.""")

ORIGINAL_H = 256
ORIGINAL_W = 448
CROP_PATCH_H = FLAGS.height
CROP_PATCH_W = FLAGS.width
GPU_ID = FLAGS.GPU_ID

import sys

sys.path.insert(1, '../tfoptflow/tfoptflow/')
from copy import deepcopy
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS

nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = '../tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = ['/device:GPU:' + GPU_ID]
nn_opts['controller'] = '/device:GPU:' + GPU_ID
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

def _read_image_random_size(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    return tf.cast(image_decoded, dtype=tf.float32) / 255.0

def _read_image_random_size_large(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    return tf.cast(image_decoded, dtype=tf.float32) / 255.0


def flow_to_img(flow):
    flow_magnitude = tf.sqrt(1e-6 + flow[..., 0] ** 2.0 + flow[..., 1] ** 2.0)
    flow_angle = tf.atan2(flow[..., 0], flow[..., 1])

    hsv_0 = ((flow_angle / np.pi) + 1.0) / 2.0
    hsv_1 = (flow_magnitude - tf.reduce_min(flow_magnitude, axis=[1, 2], keepdims=True)) / (
                1e-6 + tf.reduce_max(flow_magnitude, axis=[1, 2], keepdims=True) - tf.reduce_min(flow_magnitude,
                                                                                                 axis=[1, 2],
                                                                                                 keepdims=True))
    hsv_2 = tf.ones(tf.shape(hsv_0))
    hsv = tf.stack([hsv_0, hsv_1, hsv_2], -1)
    rgb = tf.image.hsv_to_rgb(hsv)

    return rgb
    
def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))

    grid_x = tf.reshape(tf.range(width), [1, 1, width])
    grid_x = tf.tile(grid_x, [num_batch, height, 1])
    grid_y = tf.reshape(tf.range(height), [1, height, 1])
    grid_y = tf.tile(grid_y, [num_batch, 1, width])

    flow_u, flow_v = tf.unstack(flow, 2, 3)
    pos_x = tf.cast(grid_x, dtype=tf.float32) + flow_u
    pos_y = tf.cast(grid_y, dtype=tf.float32) + flow_v
    inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                              pos_x >= 0.0)
    inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                              pos_y >= 0.0)
    inside = tf.logical_and(inside_x, inside_y)
    return tf.expand_dims(tf.cast(inside, tf.float32), 3)


def warp(I, F, H, W):
    return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)),
                      [FLAGS.batch_size, H, W, 3])

def train():
    with tf.Graph().as_default():
        def get_online_data(path):
            data_list_F0 = sorted(glob.glob(path))
            dataset_F0 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_F0))
            dataset_F0 = dataset_F0.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=30, count=None, seed=6)).map(_read_image_random_size).map(
                lambda image: tf.random_crop(image, [CROP_PATCH_H, CROP_PATCH_W, 3], seed=6))
            dataset_F0 = dataset_F0.prefetch(16)
            return dataset_F0
        def get_online_data_large(path):
            data_list_F0 = sorted(glob.glob(path))
            dataset_F0 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_F0))
            dataset_F0 = dataset_F0.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=30, count=None, seed=6)).map(_read_image_random_size_large)
            dataset_F0 = dataset_F0.prefetch(16)
            return dataset_F0

        
        """resize training images into 16x"""
        if not os.path.exists('tmp_large_image'):
            os.makedirs('tmp_large_image')
        def resize_and_save(img_path):
            original_img = cv2.imread(img_path)
            NEW_H = int(np.ceil(float(original_img.shape[0]) / 16.0)) * 16
            NEW_W = int(np.ceil(float(original_img.shape[1]) / 16.0)) * 16
            new_img = cv2.resize(original_img, dsize=(NEW_W, NEW_H), interpolation=cv2.INTER_CUBIC)
            new_path = os.path.join('tmp_large_image', os.path.split(img_path)[-1])
            cv2.imwrite(new_path, new_img)
        for img_path in sorted(glob.glob(FLAGS.training_data_path + FLAGS.training_scene + '*.png')):
            resize_and_save(img_path)
        dataset_online_I0 = get_online_data('tmp_large_image/'+FLAGS.training_scene+'*I0.png')
        dataset_online_I1 = get_online_data('tmp_large_image/'+FLAGS.training_scene+'*I1.png')
        dataset_online_I2 = get_online_data('tmp_large_image/'+FLAGS.training_scene+'*I2.png')
        dataset_online_I3 = get_online_data('tmp_large_image/'+FLAGS.training_scene+'*I3.png')
        dataset_online_I4 = get_online_data('tmp_large_image/'+FLAGS.training_scene+'*I4.png')
        batch_online_I0 = dataset_online_I0.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I1 = dataset_online_I1.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I2 = dataset_online_I2.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I3 = dataset_online_I3.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I4 = dataset_online_I4.batch(FLAGS.batch_size).make_initializable_iterator()
        fused_frame0 = batch_online_I0.get_next()
        fused_frame1 = batch_online_I1.get_next()
        fused_frame2 = batch_online_I2.get_next()
        fused_frame3 = batch_online_I3.get_next()
        fused_frame4 = batch_online_I4.get_next()
        dataset_online_I0_large = get_online_data_large('tmp_large_image/'+FLAGS.training_scene+'*I0.png')
        dataset_online_I1_large = get_online_data_large('tmp_large_image/'+FLAGS.training_scene+'*I1.png')
        dataset_online_I2_large = get_online_data_large('tmp_large_image/'+FLAGS.training_scene+'*I2.png')
        dataset_online_I3_large = get_online_data_large('tmp_large_image/'+FLAGS.training_scene+'*I3.png')
        dataset_online_I4_large = get_online_data_large('tmp_large_image/'+FLAGS.training_scene+'*I4.png')
        batch_online_I0_large = dataset_online_I0_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I1_large = dataset_online_I1_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I2_large = dataset_online_I2_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I3_large = dataset_online_I3_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I4_large = dataset_online_I4_large.batch(FLAGS.batch_size).make_initializable_iterator()
        fused_frame0_large = batch_online_I0_large.get_next()
        fused_frame1_large = batch_online_I1_large.get_next()
        fused_frame2_large = batch_online_I2_large.get_next()
        fused_frame3_large = batch_online_I3_large.get_next()
        fused_frame4_large = batch_online_I4_large.get_next()

        def PWC_full(F, B, lvl_h, lvl_w, pwc_h, pwc_w, lvl, frameNum=5):
            ratio_h = float(lvl_h) / float(pwc_h)
            ratio_w = float(lvl_w) / float(pwc_w)
            nn = ModelPWCNet(mode='test', options=nn_opts)
            nn.print_config()
            F_tmp = []
            B_tmp = []
            for i in range(frameNum):
                F_tmp.append(tf.image.resize_bilinear(F[i], (pwc_h, pwc_w), align_corners=True))
                B_tmp.append(tf.image.resize_bilinear(B[i], (pwc_h, pwc_w), align_corners=True))

            tmp_list = []
            for i in range(frameNum):
                for j in range(frameNum):
                    tmp_list.append(tf.stack([F_tmp[i], F_tmp[j]], 1))
            for i in range(frameNum):
                for j in range(frameNum):
                    tmp_list.append(tf.stack([B_tmp[i], B_tmp[j]], 1))

            PWC_input = tf.concat(tmp_list, 0)  # [batch_size*20, 2, H, W, 3]
            PWC_input = tf.reshape(PWC_input, [FLAGS.batch_size * (frameNum*frameNum*2), 2, pwc_h, pwc_w, 3])
            pred_labels, _ = nn.nn(PWC_input, reuse=tf.AUTO_REUSE)
            print(pred_labels)

            pred_labels = tf.image.resize_bilinear(pred_labels, (lvl_h, lvl_w), align_corners=True)
            """
            0: W
            1: H
            """
            ratio_tensor = tf.expand_dims(tf.expand_dims(
                tf.expand_dims(tf.convert_to_tensor(np.asarray([ratio_w, ratio_h]), dtype=tf.float32), 0), 0), 0)

            FF = []
            FB = []
            counter = 0
            for i in range(frameNum):
                FF_tmp = []
                FB_tmp = []
                for j in range(frameNum):
                    FF_tmp.append(tf.stop_gradient(pred_labels[FLAGS.batch_size * counter:FLAGS.batch_size * (counter+1)] * ratio_tensor))
                    FB_tmp.append(tf.stop_gradient(pred_labels[FLAGS.batch_size * (counter+frameNum*frameNum):FLAGS.batch_size * (counter + 1 + frameNum*frameNum)] * ratio_tensor))
                    counter += 1
                FF.append(FF_tmp)
                FB.append(FB_tmp)

            return FF, FB

        model = Decomposition_Net_Translation(CROP_PATCH_H // 16, CROP_PATCH_W // 16, False, False)
        FF01_4, FF02_4, FF03_4, FF04_4, \
        FF10_4, FF12_4, FF13_4, FF14_4, \
        FF20_4, FF21_4, FF23_4, FF24_4, \
        FF30_4, FF31_4, FF32_4, FF34_4, \
        FF40_4, FF41_4, FF42_4, FF43_4, \
        FB01_4, FB02_4, FB03_4, FB04_4, \
        FB10_4, FB12_4, FB13_4, FB14_4, \
        FB20_4, FB21_4, FB23_4, FB24_4, \
        FB30_4, FB31_4, FB32_4, FB34_4, \
        FB40_4, FB41_4, FB42_4, FB43_4 = model.inference(fused_frame0_large, fused_frame1_large, fused_frame2_large, fused_frame3_large, fused_frame4_large)

        """image"""
        model4 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=4)
        F_pred_4, B_pred_4 = model4._build_model([fused_frame0, fused_frame1, fused_frame2, fused_frame3, fused_frame4],
                                                 None, None,
                                                 [[tf.zeros_like(FF01_4), FF01_4, FF02_4, FF03_4, FF04_4],
                                                  [FF10_4, tf.zeros_like(FF10_4), FF12_4, FF13_4, FF14_4],
                                                  [FF20_4, FF21_4, tf.zeros_like(FF20_4), FF23_4, FF24_4],
                                                  [FF30_4, FF31_4, FF32_4, tf.zeros_like(FF32_4), FF34_4],
                                                  [FF40_4, FF41_4, FF42_4, FF43_4, tf.zeros_like(FF40_4)]],
                                                 [[tf.zeros_like(FB01_4), FB01_4, FB02_4, FB03_4, FB04_4],
                                                  [FB10_4, tf.zeros_like(FB10_4), FB12_4, FB13_4, FB14_4],
                                                  [FB20_4, FB21_4, tf.zeros_like(FB20_4), FB23_4, FB24_4],
                                                  [FB30_4, FB31_4, FB32_4, tf.zeros_like(FB32_4), FB34_4],
                                                  [FB40_4, FB41_4, FB42_4, FB43_4, tf.zeros_like(FB40_4)]])
        FF_3, FB_3 = PWC_full(F_pred_4, B_pred_4,
                              CROP_PATCH_H // (2 ** 4), CROP_PATCH_W // (2 ** 4),
                              int(np.ceil(float(CROP_PATCH_H // (2 ** 4)) / 64.0)) * 64,
                              int(np.ceil(float(CROP_PATCH_W // (2 ** 4)) / 64.0)) * 64, 3)

        model3 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=3)
        F_pred_3, B_pred_3 = model3._build_model([fused_frame0, fused_frame1, fused_frame2, fused_frame3, fused_frame4],
                                                 F_pred_4, B_pred_4, FF_3, FB_3)

        FF_2, FB_2 = PWC_full(F_pred_3, B_pred_3,
                              CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3),
                              int(np.ceil(float(CROP_PATCH_H // (2 ** 3)) / 64.0)) * 64,
                              int(np.ceil(float(CROP_PATCH_W // (2 ** 3)) / 64.0)) * 64, 2)

        model2 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=2)
        F_pred_2, B_pred_2 = model2._build_model([fused_frame0, fused_frame1, fused_frame2, fused_frame3, fused_frame4],
                                                 F_pred_3, B_pred_3, FF_2, FB_2)
        FF_1, FB_1 = PWC_full(F_pred_2, B_pred_2,
                              CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2),
                              int(np.ceil(float(CROP_PATCH_H // (2 ** 2)) / 64.0)) * 64,
                              int(np.ceil(float(CROP_PATCH_W // (2 ** 2)) / 64.0)) * 64, 1)

        model1 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=1)
        F_pred_1, B_pred_1 = model1._build_model([fused_frame0, fused_frame1, fused_frame2, fused_frame3, fused_frame4],
                                                 F_pred_2, B_pred_2, FF_1, FB_1)
        FF_0, FB_0 = PWC_full(F_pred_1, B_pred_1,
                              CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1),
                              int(np.ceil(float(CROP_PATCH_H // (2 ** 1)) / 64.0)) * 64,
                              int(np.ceil(float(CROP_PATCH_W // (2 ** 1)) / 64.0)) * 64, 0)

        model0 = ImageReconstruction_reflection_arbitraryFrameNum(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=0)
        F_pred_0, B_pred_0 = model0._build_model([fused_frame0, fused_frame1, fused_frame2, fused_frame3, fused_frame4],
                                                 F_pred_1, B_pred_1, FF_0, FB_0)
                                                 
        """clip to 0 1"""
        for i in range(5):
            F_pred_0[i] = tf.clip_by_value(F_pred_0[i], 0.0, 1.0)
            F_pred_1[i] = tf.clip_by_value(F_pred_1[i], 0.0, 1.0)
            F_pred_2[i] = tf.clip_by_value(F_pred_2[i], 0.0, 1.0)
            F_pred_3[i] = tf.clip_by_value(F_pred_3[i], 0.0, 1.0)
            F_pred_4[i] = tf.clip_by_value(F_pred_4[i], 0.0, 1.0)
            B_pred_0[i] = tf.clip_by_value(B_pred_0[i], 0.0, 1.0)
            B_pred_1[i] = tf.clip_by_value(B_pred_1[i], 0.0, 1.0)
            B_pred_2[i] = tf.clip_by_value(B_pred_2[i], 0.0, 1.0)
            B_pred_3[i] = tf.clip_by_value(B_pred_3[i], 0.0, 1.0)
            B_pred_4[i] = tf.clip_by_value(B_pred_4[i], 0.0, 1.0)

        """blur image and compute PWC"""
        F_blurred = []
        B_blurred = []
        def generate_gaussian_kernel(sz):
            kernel = cv2.getGaussianKernel(sz, 0)
            kernel = np.dot(kernel, kernel.transpose())
            return tf.cast(kernel[:, :, np.newaxis, np.newaxis], tf.float32)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        kernel = generate_gaussian_kernel(21)
        def apply_gaussian_blur_image(x):
            x = tf.pad(x, [[0, 0], [40, 40], [40, 40], [0, 0]], 'SYMMETRIC')
            x_0 = tf.nn.conv2d(x[..., 0:1], kernel, strides=[1, 1, 1, 1], padding="SAME")
            x_1 = tf.nn.conv2d(x[..., 1:2], kernel, strides=[1, 1, 1, 1], padding="SAME")
            x_2 = tf.nn.conv2d(x[..., 2:3], kernel, strides=[1, 1, 1, 1], padding="SAME")
            output = tf.concat([x_0, x_1, x_2], -1)
            return output[:, 40:-40, 40:-40]
        F_max = tf.reduce_max(tf.concat([F_pred_0[0], F_pred_0[1], F_pred_0[2], F_pred_0[3], F_pred_0[4]], -1), keepdims=True)
        F_min = tf.reduce_min(tf.concat([F_pred_0[0], F_pred_0[1], F_pred_0[2], F_pred_0[3], F_pred_0[4]], -1), keepdims=True)
        B_max = tf.reduce_max(tf.concat([B_pred_0[0], B_pred_0[1], B_pred_0[2], B_pred_0[3], B_pred_0[4]], -1), keepdims=True)
        B_min = tf.reduce_min(tf.concat([B_pred_0[0], B_pred_0[1], B_pred_0[2], B_pred_0[3], B_pred_0[4]], -1), keepdims=True)
        for i in range(len(F_pred_0)):
            tmp_F = apply_gaussian_blur_image(F_pred_0[i])
            tmp_B = apply_gaussian_blur_image(B_pred_0[i])
            F_blurred.append((tmp_F - F_min) / (F_max - F_min + 1e-6))
            B_blurred.append((tmp_B - B_min) / (B_max - B_min + 1e-6))

        FF_full, FB_full = PWC_full(F_blurred, B_blurred,
                              CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0),
                              int(np.ceil(float(CROP_PATCH_H // (2 ** 0)) / 64.0)) * 64,
                              int(np.ceil(float(CROP_PATCH_W // (2 ** 0)) / 64.0)) * 64, 0)

        """loss"""
        F_pred = []
        F_pred.append(tf.concat([F_pred_0[0], F_pred_0[1], F_pred_0[2], F_pred_0[3], F_pred_0[4]], -1))
        F_pred.append(tf.concat([F_pred_1[0], F_pred_1[1], F_pred_1[2], F_pred_1[3], F_pred_1[4]], -1))
        F_pred.append(tf.concat([F_pred_2[0], F_pred_2[1], F_pred_2[2], F_pred_2[3], F_pred_2[4]], -1))
        F_pred.append(tf.concat([F_pred_3[0], F_pred_3[1], F_pred_3[2], F_pred_3[3], F_pred_3[4]], -1))
        F_pred.append(tf.concat([F_pred_4[0], F_pred_4[1], F_pred_4[2], F_pred_4[3], F_pred_4[4]], -1))
        B_pred = []
        B_pred.append(tf.concat([B_pred_0[0], B_pred_0[1], B_pred_0[2], B_pred_0[3], B_pred_0[4]], -1))
        B_pred.append(tf.concat([B_pred_1[0], B_pred_1[1], B_pred_1[2], B_pred_1[3], B_pred_1[4]], -1))
        B_pred.append(tf.concat([B_pred_2[0], B_pred_2[1], B_pred_2[2], B_pred_2[3], B_pred_2[4]], -1))
        B_pred.append(tf.concat([B_pred_3[0], B_pred_3[1], B_pred_3[2], B_pred_3[3], B_pred_3[4]], -1))
        B_pred.append(tf.concat([B_pred_4[0], B_pred_4[1], B_pred_4[2], B_pred_4[3], B_pred_4[4]], -1))
        """finest level flow"""
        FF_ = []
        FB_ = []
        for i in range(len(FF_0)):
            FF_sub = []
            FB_sub = []
            for j in range(len(FF_0[i])):

                FF_sub.append(FF_full[i][j])
                FB_sub.append(FB_full[i][j])
                
            FF_.append(FF_sub)
            FB_.append(FB_sub)


        loss = 0
        loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(1):
            h = int(CROP_PATCH_H // (2**i))
            w = int(CROP_PATCH_W // (2**i))
            print('level: ' + str(i))
            print(h)
            print(w)
            I0_lvl = fused_frame0
            I1_lvl = fused_frame1
            I2_lvl = fused_frame2
            I3_lvl = fused_frame3
            I4_lvl = fused_frame4

            def compute_loss_2(FF02, FB02, FF12, FB12, FF32, FB32, FF42, FB42, F2, B2, I2_lvl, I0_lvl, I1_lvl, I3_lvl,
                               I4_lvl):
                sub_loss = 0

                """resize predictions"""
                F2 = tf.image.resize_bilinear(F2, [CROP_PATCH_H, CROP_PATCH_W], align_corners=True)
                B2 = tf.image.resize_bilinear(B2, [CROP_PATCH_H, CROP_PATCH_W], align_corners=True)

                """compute loss only in (mask1 & mask2)"""
                outmask = create_outgoing_mask(FF02) * create_outgoing_mask(FB02)
                sub_loss += (loss_weight[i] * tf.reduce_sum(tf.abs(I0_lvl - warp(F2, FF02, CROP_PATCH_H, CROP_PATCH_W) - warp(B2, FB02, CROP_PATCH_H, CROP_PATCH_W)) * outmask) / (3*tf.reduce_sum(outmask)+1e-10))
                outmask = create_outgoing_mask(FF12) * create_outgoing_mask(FB12)
                sub_loss += (loss_weight[i] * tf.reduce_sum(tf.abs(I1_lvl - warp(F2, FF12, CROP_PATCH_H, CROP_PATCH_W) - warp(B2, FB12, CROP_PATCH_H, CROP_PATCH_W)) * outmask) / (3*tf.reduce_sum(outmask)+1e-10))
                sub_loss += (loss_weight[i] * tf.reduce_mean(tf.abs(I2_lvl - F2 - B2)))
                outmask = create_outgoing_mask(FF32) * create_outgoing_mask(FB32)
                sub_loss += (loss_weight[i] * tf.reduce_sum(tf.abs(I3_lvl - warp(F2, FF32, CROP_PATCH_H, CROP_PATCH_W) - warp(B2, FB32, CROP_PATCH_H, CROP_PATCH_W)) * outmask) / (3*tf.reduce_sum(outmask)+1e-10))
                outmask = create_outgoing_mask(FF42) * create_outgoing_mask(FB42)
                sub_loss += (loss_weight[i] * tf.reduce_sum(tf.abs(I4_lvl - warp(F2, FF42, CROP_PATCH_H, CROP_PATCH_W) - warp(B2, FB42, CROP_PATCH_H, CROP_PATCH_W)) * outmask) / (3*tf.reduce_sum(outmask)+1e-10))

                """TV loss"""
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(F2[:, 1:] - F2[:, :-1]))))
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(F2[:, :, 1:] - F2[:, :, :-1]))))
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(B2[:, 1:] - B2[:, :-1]))))
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(B2[:, :, 1:] - B2[:, :, :-1]))))
                return sub_loss

            """full size PWC"""
            loss += compute_loss_2(FF_[0][2], FB_[0][2], FF_[1][2], FB_[1][2], FF_[3][2], FB_[3][2], FF_[4][2], FB_[4][2],
                                   tf.clip_by_value(F_pred[i][..., 6:9], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 6:9], 0.0, 1.0),
                                   I2_lvl, I0_lvl, I1_lvl, I3_lvl, I4_lvl)
            loss += compute_loss_2(FF_[1][0], FB_[1][0], FF_[2][0], FB_[2][0], FF_[3][0], FB_[3][0], FF_[4][0], FB_[4][0],
                                   tf.clip_by_value(F_pred[i][..., 0:3], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 0:3], 0.0, 1.0),
                                   I0_lvl, I1_lvl, I2_lvl, I3_lvl, I4_lvl)
            loss += compute_loss_2(FF_[0][1], FB_[0][1], FF_[2][1], FB_[2][1], FF_[3][1], FB_[3][1], FF_[4][1], FB_[4][1],
                                   tf.clip_by_value(F_pred[i][..., 3:6], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 3:6], 0.0, 1.0),
                                   I1_lvl, I0_lvl, I2_lvl, I3_lvl, I4_lvl)
            loss += compute_loss_2(FF_[0][3], FB_[0][3], FF_[1][3], FB_[1][3], FF_[2][3], FB_[2][3], FF_[4][3], FB_[4][3],
                                   tf.clip_by_value(F_pred[i][..., 9:12], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 9:12], 0.0, 1.0),
                                   I3_lvl, I0_lvl, I1_lvl, I2_lvl, I4_lvl)
            loss += compute_loss_2(FF_[0][4], FB_[0][4], FF_[1][4], FB_[1][4], FF_[2][4], FB_[2][4], FF_[3][4], FB_[3][4],
                                   tf.clip_by_value(F_pred[i][..., 12:15], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 12:15], 0.0, 1.0),
                                   I4_lvl, I0_lvl, I1_lvl, I2_lvl, I3_lvl)

        t_vars = tf.all_variables()
        print('all layers:')
        for var in t_vars: print(var.name)
        dof_vars = [var for var in t_vars if 'FusionLayer_' in var.name]
        print('optimize layers:')
        for var in dof_vars: print(var.name)

        total_parameters = 0
        for variable in tf.trainable_variables():
            if 'FusionLayer_' in variable.name:
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print(shape)
                print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    print(dim)
                    variable_parameters *= dim.value
                print(variable_parameters)
                total_parameters += variable_parameters
        print(total_parameters)

        # Create an optimizer that performs gradient descent.
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            update_op = tf.train.AdamOptimizer(FLAGS.initial_learning_rate).minimize(loss, var_list=dof_vars, global_step=global_step)

        tf.summary.scalar('loss', loss)
        tf.summary.image('fused_frame0', fused_frame0, 3)
        tf.summary.image('fused_frame1', fused_frame1, 3)
        tf.summary.image('fused_frame2', fused_frame2, 3)
        tf.summary.image('fused_frame3', fused_frame3, 3)
        tf.summary.image('fused_frame4', fused_frame4, 3)
        
        tf.summary.image('FF02', flow_to_img(FF_[0][2]), 3)
        tf.summary.image('FB02', flow_to_img(FB_[0][2]), 3)
        
        tf.summary.image('blur_F2', F_blurred[2], 3)
        tf.summary.image('blur_B2', B_blurred[2], 3)
        tf.summary.image('blur_F0', F_blurred[0], 3)
        tf.summary.image('blur_B0', B_blurred[0], 3)

        tf.summary.image('warpF0to2', warp(F_pred_0[2], FF_[0][2], CROP_PATCH_H, CROP_PATCH_W), 3)
        tf.summary.image('warpB0to2', warp(B_pred_0[2], FB_[0][2], CROP_PATCH_H, CROP_PATCH_W), 3)
        tf.summary.image('sum0to2', tf.clip_by_value(warp(F_pred_0[2], FF_[0][2], CROP_PATCH_H, CROP_PATCH_W) + warp(B_pred_0[2], FB_[0][2], CROP_PATCH_H, CROP_PATCH_W), 0.0, 1.0), 3)
        tf.summary.image('sum2', tf.clip_by_value(F_pred_0[2] + B_pred_0[2], 0.0 ,1.0), 3)

        tf.summary.image('F2_pred_0', F_pred_0[2], 3)
        tf.summary.image('B2_pred_0', B_pred_0[2], 3)
        tf.summary.image('F0_pred_0', F_pred_0[0], 3)
        tf.summary.image('B0_pred_0', B_pred_0[0], 3)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

        # Restore checkpoint from file.
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run([init,
                  batch_online_I0.initializer, batch_online_I1.initializer, batch_online_I2.initializer,
                  batch_online_I3.initializer, batch_online_I4.initializer,
                  batch_online_I0_large.initializer, batch_online_I1_large.initializer,
                  batch_online_I2_large.initializer, batch_online_I3_large.initializer,
                  batch_online_I4_large.initializer])

        saver2 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "pwcnet" in v.name])
        saver2.restore(sess, nn_opts['ckpt_path'])
        saver4 = tf.train.Saver(var_list=[v for v in tf.all_variables() if
                                          "FeaturePyramidExtractor" in v.name or "TranslationEstimator" in v.name])
        saver4.restore(sess, 'train_dir_initFlow_Reflection/model.ckpt-239999')
        saver5 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "FusionLayer_" in v.name])
        saver5.restore(sess, 'train_dir_imgReconstruction_Reflection/model.ckpt-239999')

        # Summary Writter
        summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir,
            graph=sess.graph)


        for step in range(0, FLAGS.max_steps):

            # Run single step update.
            _, loss_value = sess.run([update_op, loss])

            if step % 10 == 0:
                print("Loss at step %d: %f" % (step, loss_value))

            if step % 10 == 0:
                # Output Summary
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save checkpoint
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    train()
