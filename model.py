from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from warp_utils import dense_image_warp

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001

from functools import partial

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

class Decomposition_Net_Translation(object):
    def __init__(self, H, W, is_training, use_BN=False):
        self.lvl = 4
        self.filters = [16, 32, 64, 96]
        self.s_range = 4
        self.H = H
        self.W = W
        self.is_training = is_training
        self.use_BN = use_BN

    def inference(self, I0, I1, I2, I3, I4):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(I0, I1, I2, I3, I4)

    def FeaturePyramidExtractor(self, x):
        with tf.variable_scope("FeaturePyramidExtractor", reuse=tf.AUTO_REUSE):
            for l in range(self.lvl):
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
            return x

    def CostVolumeLayer(self, features_0, features_0from1):
        cost_length = (2 * self.s_range + 1) ** 2

        def get_cost(features_0, features_0from1, shift):
            def pad2d(x, vpad, hpad):
                return tf.pad(x, [[0, 0], vpad, hpad, [0, 0]])

            def crop2d(x, vcrop, hcrop):
                return tf.keras.layers.Cropping2D([vcrop, hcrop])(x)

            """
            Calculate cost volume for specific shift
            - inputs
            features_0 (batch, h, w, nch): feature maps at time slice 0
            features_0from1 (batch, h, w, nch): feature maps at time slice 0 warped from 1
            shift (2): spatial (vertical and horizontal) shift to be considered
            - output
            cost (batch, h, w): cost volume map for the given shift
            """
            v, h = shift  # vertical/horizontal element
            vt, vb, hl, hr = max(v, 0), abs(min(v, 0)), max(h, 0), abs(min(h, 0))  # top/bottom left/right
            f_0_pad = pad2d(features_0, [vt, vb], [hl, hr])
            f_0from1_pad = pad2d(features_0from1, [vb, vt], [hr, hl])
            cost_pad = f_0_pad * f_0from1_pad
            return tf.reduce_mean(crop2d(cost_pad, [vt, vb], [hl, hr]), axis=3)

        get_c = partial(get_cost, features_0, features_0from1)
        cv = [0] * cost_length
        depth = 0
        for v in range(-self.s_range, self.s_range + 1):
            for h in range(-self.s_range, self.s_range + 1):
                cv[depth] = get_c(shift=[v, h])
                depth += 1

        cv = tf.stack(cv, axis=3)
        cv = tf.nn.leaky_relu(cv, 0.1)
        return cv

    def TranslationEstimator(self, feature_2, feature_0):
        def _conv_block(filters, kernel_size=(3, 3), strides=(1, 1)):
            def f(x):
                x = tf.layers.Conv2D(filters, kernel_size, strides, 'valid')(x)
                x = tf.nn.leaky_relu(x, 0.2)
                return x

            return f

        with tf.variable_scope("TranslationEstimator", reuse=tf.AUTO_REUSE):
            cost = self.CostVolumeLayer(feature_2, feature_0)
            x = tf.concat([feature_2, cost], axis=3)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(96, (3, 3), (1, 1))(x)
            x = _conv_block(64, (3, 3), (1, 1))(x)
            feature = _conv_block(32, (3, 3), (1, 1))(x)
            x = tf.reduce_mean(feature, axis=[1, 2])
            flow1 = tf.layers.dense(x, 2)
            flow2 = tf.layers.dense(x, 2)
            flow1 = tf.expand_dims(tf.expand_dims(flow1, 1), 1)
            flow2 = tf.expand_dims(tf.expand_dims(flow2, 1), 1)
            # flow1 = tf.tile(flow1, [1, feature_2.get_shape().as_list()[1], feature_2.get_shape().as_list()[2], 1])
            # flow2 = tf.tile(flow2, [1, feature_2.get_shape().as_list()[1], feature_2.get_shape().as_list()[2], 1])
            flow1 = tf.tile(flow1, [1, self.H, self.W, 1])
            flow2 = tf.tile(flow2, [1, self.H, self.W, 1])
            return flow1, flow2

    def HomographyEstimator(self, feature_2, feature_0):
        def _conv_block(filters, kernel_size=(3, 3), strides=(1, 1)):
            def f(x):
                x = tf.layers.Conv2D(filters, kernel_size, strides, 'same')(x)
                if self.use_BN:
                    x = tf.layers.batch_normalization(x, training=self.is_training, trainable=self.is_training)
                x = tf.nn.leaky_relu(x, 0.2)
                return x
            return f

        def homography_matrix_to_flow(tf_homography_matrix, im_shape_w, im_shape_h):
            # tf_homography_matrix [B, 3, 3]
            import numpy as np
            grid_x, grid_y = tf.meshgrid(tf.range(im_shape_w), tf.range(im_shape_h))
            if not self.is_training:
                grid_x = tf.cast(grid_x, tf.float32) / tf.convert_to_tensor(float(self.W)) * tf.convert_to_tensor(20.0)
                grid_y = tf.cast(grid_y, tf.float32) / tf.convert_to_tensor(float(self.H)) * tf.convert_to_tensor(12.0)

            grid_z = tf.ones_like(grid_x)
            tf_XYZ = tf.cast(tf.stack([grid_y, grid_x, grid_z], axis=-1), tf.float32)

            # Y, X = np.meshgrid(range(im_shape_w), range(im_shape_h))
            # Z = np.ones_like(X)
            # XYZ = np.stack((Y, X, Z), axis=-1)
            # tf_XYZ = tf.constant(XYZ.astype("float32"))
            tf_XYZ = tf_XYZ[tf.newaxis, :, :, :, tf.newaxis]  # [1, H, W, 3, 1]
            tf_XYZ = tf.tile(tf_XYZ, [tf_homography_matrix.get_shape().as_list()[0], 1, 1, 1, 1])  # [B, H, W, 3, 1]
            tf_homography_matrix = tf.tile(tf_homography_matrix[:, tf.newaxis, tf.newaxis], (1, im_shape_h, im_shape_w, 1, 1))  # [B, H, W, 3, 3]
            tf_unnormalized_transformed_XYZ = tf.matmul(tf_homography_matrix, tf_XYZ, transpose_b=False)  # [B, H, W, 3, 1]
            tf_transformed_XYZ = tf_unnormalized_transformed_XYZ / tf_unnormalized_transformed_XYZ[:, :, :, -1][:, :, :, tf.newaxis, :]
            flow = -(tf_transformed_XYZ - tf_XYZ)[..., :2, 0]

            if not self.is_training:
                ratio_h = float(self.H) / 12.0
                ratio_w = float(self.W) / 20.0
                ratio_tensor = tf.expand_dims(tf.expand_dims(
                    tf.expand_dims(tf.convert_to_tensor(np.asarray([ratio_w, ratio_h]), dtype=tf.float32), 0), 0), 0)
                flow = flow * ratio_tensor

            return flow

        with tf.variable_scope("HomographyEstimator", reuse=tf.AUTO_REUSE):
            cost = self.CostVolumeLayer(feature_2, feature_0)
            grid_x, grid_y = tf.meshgrid(tf.range(self.W), tf.range(self.H))
            grid_x = tf.cast(grid_x, tf.float32) / (tf.ones([1, 1])*self.W)
            grid_y = tf.cast(grid_y, tf.float32) / (tf.ones([1, 1])*self.H)
            grid_x = tf.tile(tf.expand_dims(tf.expand_dims(grid_x, 0), -1), [feature_2.get_shape().as_list()[0], 1, 1, 1])
            grid_y = tf.tile(tf.expand_dims(tf.expand_dims(grid_y, 0), -1), [feature_2.get_shape().as_list()[0], 1, 1, 1])
            x = tf.concat([feature_2, cost, grid_x, grid_y], axis=3)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(96, (3, 3), (1, 1))(x)
            x = _conv_block(64, (3, 3), (1, 1))(x)
            feature = _conv_block(32, (3, 3), (1, 1))(x)
            x = tf.reduce_mean(feature, axis=[1, 2])
            flow1 = tf.layers.dense(x, 8)
            flow1 = tf.concat([flow1, tf.zeros([flow1.get_shape().as_list()[0], 1], tf.float32)], -1)
            # flow1 = tf.concat([flow1, tf.ones([flow1.get_shape().as_list()[0], 1], tf.float32)], -1)
            flow1 = tf.reshape(flow1, [flow1.get_shape().as_list()[0], 3, 3])
            flow1 = tf.eye(3, 3, [flow1.get_shape().as_list()[0]]) + flow1
            flow1 = homography_matrix_to_flow(flow1, self.W, self.H)

            flow2 = tf.layers.dense(x, 8)
            flow2 = tf.concat([flow2, tf.zeros([flow2.get_shape().as_list()[0], 1], tf.float32)], -1)
            # flow2 = tf.concat([flow2, tf.ones([flow2.get_shape().as_list()[0], 1], tf.float32)], -1)
            flow2 = tf.reshape(flow2, [flow2.get_shape().as_list()[0], 3, 3])
            flow2 = tf.eye(3, 3, [flow2.get_shape().as_list()[0]]) + flow2
            flow2 = homography_matrix_to_flow(flow2, self.W, self.H)
            return flow1, flow2

    def warp(self, I, F, b, h, w, c):
        return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [b, h, w, c])

    def _build_model(self, image_0, image_1, image_2, image_3, image_4):

        """P"""
        feature_0 = self.FeaturePyramidExtractor(image_0)
        feature_1 = self.FeaturePyramidExtractor(image_1)
        feature_2 = self.FeaturePyramidExtractor(image_2)
        feature_3 = self.FeaturePyramidExtractor(image_3)
        feature_4 = self.FeaturePyramidExtractor(image_4)

        Estimator = self.TranslationEstimator

        FF01, FB01 = Estimator(feature_0, feature_1)
        FF02, FB02 = Estimator(feature_0, feature_2)
        FF03, FB03 = Estimator(feature_0, feature_3)
        FF04, FB04 = Estimator(feature_0, feature_4)

        FF10, FB10 = Estimator(feature_1, feature_0)
        FF12, FB12 = Estimator(feature_1, feature_2)
        FF13, FB13 = Estimator(feature_1, feature_3)
        FF14, FB14 = Estimator(feature_1, feature_4)

        FF20, FB20 = Estimator(feature_2, feature_0)
        FF21, FB21 = Estimator(feature_2, feature_1)
        FF23, FB23 = Estimator(feature_2, feature_3)
        FF24, FB24 = Estimator(feature_2, feature_4)

        FF30, FB30 = Estimator(feature_3, feature_0)
        FF31, FB31 = Estimator(feature_3, feature_1)
        FF32, FB32 = Estimator(feature_3, feature_2)
        FF34, FB34 = Estimator(feature_3, feature_4)

        FF40, FB40 = Estimator(feature_4, feature_0)
        FF41, FB41 = Estimator(feature_4, feature_1)
        FF42, FB42 = Estimator(feature_4, feature_2)
        FF43, FB43 = Estimator(feature_4, feature_3)

        return FF01, FF02, FF03, FF04, \
               FF10, FF12, FF13, FF14, \
               FF20, FF21, FF23, FF24, \
               FF30, FF31, FF32, FF34, \
               FF40, FF41, FF42, FF43, \
               FB01, FB02, FB03, FB04, \
               FB10, FB12, FB13, FB14, \
               FB20, FB21, FB23, FB24, \
               FB30, FB31, FB32, FB34, \
               FB40, FB41, FB42, FB43

class Decomposition_Net_Translation_arbitraryFrameNum(object):
    def __init__(self, H, W, is_training, use_BN=False):
        self.lvl = 4
        self.filters = [16, 32, 64, 96]
        self.s_range = 4
        self.H = H
        self.W = W
        self.is_training = is_training
        self.use_BN = use_BN

    def inference(self, I):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(I)

    def FeaturePyramidExtractor(self, x):
        with tf.variable_scope("FeaturePyramidExtractor", reuse=tf.AUTO_REUSE):
            for l in range(self.lvl):
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
            return x

    def CostVolumeLayer(self, features_0, features_0from1):
        cost_length = (2 * self.s_range + 1) ** 2

        def get_cost(features_0, features_0from1, shift):
            def pad2d(x, vpad, hpad):
                return tf.pad(x, [[0, 0], vpad, hpad, [0, 0]])

            def crop2d(x, vcrop, hcrop):
                return tf.keras.layers.Cropping2D([vcrop, hcrop])(x)

            """
            Calculate cost volume for specific shift
            - inputs
            features_0 (batch, h, w, nch): feature maps at time slice 0
            features_0from1 (batch, h, w, nch): feature maps at time slice 0 warped from 1
            shift (2): spatial (vertical and horizontal) shift to be considered
            - output
            cost (batch, h, w): cost volume map for the given shift
            """
            v, h = shift  # vertical/horizontal element
            vt, vb, hl, hr = max(v, 0), abs(min(v, 0)), max(h, 0), abs(min(h, 0))  # top/bottom left/right
            f_0_pad = pad2d(features_0, [vt, vb], [hl, hr])
            f_0from1_pad = pad2d(features_0from1, [vb, vt], [hr, hl])
            cost_pad = f_0_pad * f_0from1_pad
            return tf.reduce_mean(crop2d(cost_pad, [vt, vb], [hl, hr]), axis=3)

        get_c = partial(get_cost, features_0, features_0from1)
        cv = [0] * cost_length
        depth = 0
        for v in range(-self.s_range, self.s_range + 1):
            for h in range(-self.s_range, self.s_range + 1):
                cv[depth] = get_c(shift=[v, h])
                depth += 1

        cv = tf.stack(cv, axis=3)
        cv = tf.nn.leaky_relu(cv, 0.1)
        return cv

    def TranslationEstimator(self, feature_2, feature_0):
        def _conv_block(filters, kernel_size=(3, 3), strides=(1, 1)):
            def f(x):
                x = tf.layers.Conv2D(filters, kernel_size, strides, 'valid')(x)
                x = tf.nn.leaky_relu(x, 0.2)
                return x

            return f

        with tf.variable_scope("TranslationEstimator", reuse=tf.AUTO_REUSE):
            cost = self.CostVolumeLayer(feature_2, feature_0)
            x = tf.concat([feature_2, cost], axis=3)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(96, (3, 3), (1, 1))(x)
            x = _conv_block(64, (3, 3), (1, 1))(x)
            feature = _conv_block(32, (3, 3), (1, 1))(x)
            x = tf.reduce_mean(feature, axis=[1, 2])
            flow1 = tf.layers.dense(x, 2)
            flow2 = tf.layers.dense(x, 2)
            flow1 = tf.expand_dims(tf.expand_dims(flow1, 1), 1)
            flow2 = tf.expand_dims(tf.expand_dims(flow2, 1), 1)
            # flow1 = tf.tile(flow1, [1, feature_2.get_shape().as_list()[1], feature_2.get_shape().as_list()[2], 1])
            # flow2 = tf.tile(flow2, [1, feature_2.get_shape().as_list()[1], feature_2.get_shape().as_list()[2], 1])
            flow1 = tf.tile(flow1, [1, self.H, self.W, 1])
            flow2 = tf.tile(flow2, [1, self.H, self.W, 1])
            return flow1, flow2

    def HomographyEstimator(self, feature_2, feature_0):
        def _conv_block(filters, kernel_size=(3, 3), strides=(1, 1)):
            def f(x):
                x = tf.layers.Conv2D(filters, kernel_size, strides, 'same')(x)
                if self.use_BN:
                    x = tf.layers.batch_normalization(x, training=self.is_training, trainable=self.is_training)
                x = tf.nn.leaky_relu(x, 0.2)
                return x
            return f

        def homography_matrix_to_flow(tf_homography_matrix, im_shape_w, im_shape_h):
            # tf_homography_matrix [B, 3, 3]
            import numpy as np
            grid_x, grid_y = tf.meshgrid(tf.range(im_shape_w), tf.range(im_shape_h))
            if not self.is_training:
                grid_x = tf.cast(grid_x, tf.float32) / tf.convert_to_tensor(float(self.W)) * tf.convert_to_tensor(20.0)
                grid_y = tf.cast(grid_y, tf.float32) / tf.convert_to_tensor(float(self.H)) * tf.convert_to_tensor(12.0)

            grid_z = tf.ones_like(grid_x)
            tf_XYZ = tf.cast(tf.stack([grid_y, grid_x, grid_z], axis=-1), tf.float32)

            # Y, X = np.meshgrid(range(im_shape_w), range(im_shape_h))
            # Z = np.ones_like(X)
            # XYZ = np.stack((Y, X, Z), axis=-1)
            # tf_XYZ = tf.constant(XYZ.astype("float32"))
            tf_XYZ = tf_XYZ[tf.newaxis, :, :, :, tf.newaxis]  # [1, H, W, 3, 1]
            tf_XYZ = tf.tile(tf_XYZ, [tf_homography_matrix.get_shape().as_list()[0], 1, 1, 1, 1])  # [B, H, W, 3, 1]
            tf_homography_matrix = tf.tile(tf_homography_matrix[:, tf.newaxis, tf.newaxis], (1, im_shape_h, im_shape_w, 1, 1))  # [B, H, W, 3, 3]
            tf_unnormalized_transformed_XYZ = tf.matmul(tf_homography_matrix, tf_XYZ, transpose_b=False)  # [B, H, W, 3, 1]
            tf_transformed_XYZ = tf_unnormalized_transformed_XYZ / tf_unnormalized_transformed_XYZ[:, :, :, -1][:, :, :, tf.newaxis, :]
            flow = -(tf_transformed_XYZ - tf_XYZ)[..., :2, 0]

            if not self.is_training:
                ratio_h = float(self.H) / 12.0
                ratio_w = float(self.W) / 20.0
                ratio_tensor = tf.expand_dims(tf.expand_dims(
                    tf.expand_dims(tf.convert_to_tensor(np.asarray([ratio_w, ratio_h]), dtype=tf.float32), 0), 0), 0)
                flow = flow * ratio_tensor

            return flow

        with tf.variable_scope("HomographyEstimator", reuse=tf.AUTO_REUSE):
            cost = self.CostVolumeLayer(feature_2, feature_0)
            grid_x, grid_y = tf.meshgrid(tf.range(self.W), tf.range(self.H))
            grid_x = tf.cast(grid_x, tf.float32) / (tf.ones([1, 1])*self.W)
            grid_y = tf.cast(grid_y, tf.float32) / (tf.ones([1, 1])*self.H)
            grid_x = tf.tile(tf.expand_dims(tf.expand_dims(grid_x, 0), -1), [feature_2.get_shape().as_list()[0], 1, 1, 1])
            grid_y = tf.tile(tf.expand_dims(tf.expand_dims(grid_y, 0), -1), [feature_2.get_shape().as_list()[0], 1, 1, 1])
            x = tf.concat([feature_2, cost, grid_x, grid_y], axis=3)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(96, (3, 3), (1, 1))(x)
            x = _conv_block(64, (3, 3), (1, 1))(x)
            feature = _conv_block(32, (3, 3), (1, 1))(x)
            x = tf.reduce_mean(feature, axis=[1, 2])
            flow1 = tf.layers.dense(x, 8)
            flow1 = tf.concat([flow1, tf.zeros([flow1.get_shape().as_list()[0], 1], tf.float32)], -1)
            # flow1 = tf.concat([flow1, tf.ones([flow1.get_shape().as_list()[0], 1], tf.float32)], -1)
            flow1 = tf.reshape(flow1, [flow1.get_shape().as_list()[0], 3, 3])
            flow1 = tf.eye(3, 3, [flow1.get_shape().as_list()[0]]) + flow1
            flow1 = homography_matrix_to_flow(flow1, self.W, self.H)

            flow2 = tf.layers.dense(x, 8)
            flow2 = tf.concat([flow2, tf.zeros([flow2.get_shape().as_list()[0], 1], tf.float32)], -1)
            # flow2 = tf.concat([flow2, tf.ones([flow2.get_shape().as_list()[0], 1], tf.float32)], -1)
            flow2 = tf.reshape(flow2, [flow2.get_shape().as_list()[0], 3, 3])
            flow2 = tf.eye(3, 3, [flow2.get_shape().as_list()[0]]) + flow2
            flow2 = homography_matrix_to_flow(flow2, self.W, self.H)
            return flow1, flow2

    def warp(self, I, F, b, h, w, c):
        return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [b, h, w, c])

    def _build_model(self, image):

        """P"""
        feature = []
        for I in image:
            feature.append(self.FeaturePyramidExtractor(I))

        Estimator = self.TranslationEstimator

        FF = []
        FB = []
        for i in range(len(feature)):
            FF_sub = []
            FB_sub = []
            for j in range(len(feature)):
                if i != j:
                    F_tmp, B_tmp = Estimator(feature[i], feature[j])
                    FF_sub.append(F_tmp)
                    FB_sub.append(B_tmp)
                else:
                    FF_sub.append(tf.zeros([image[0].get_shape().as_list()[0], self.H, self.W, 2]))
                    FB_sub.append(tf.zeros([image[0].get_shape().as_list()[0], self.H, self.W, 2]))
            FF.append(FF_sub)
            FB.append(FB_sub)

        return FF, FB

class ImageReconstruction_reflection_arbitraryFrameNum_large_FBconcat_AvgMeanPool(object):
    def __init__(self, batch_size, CROP_PATCH_H, CROP_PATCH_W, level=4):
        self.batch_size = batch_size
        self.CROP_PATCH_H = CROP_PATCH_H
        self.CROP_PATCH_W = CROP_PATCH_W
        self.level = level

    def sub_net(self, x):
        with tf.variable_scope("sub_net", reuse=tf.AUTO_REUSE):
            x = tf.nn.elu(tf.layers.conv2d(x, 32, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            return x

    def FusionLayer_F(self, image_2_F, image_2_B, key_frame, I_list, F_list, lvl):
        with tf.variable_scope("FusionLayer_F_" + str(lvl), reuse=tf.AUTO_REUSE):
            b, h, w, _ = tf.unstack(tf.shape(image_2_F))

            F_registrated = []
            for i in range(len(I_list)):
                registrated_foreground = self.warp(I_list[i], F_list[i], b, h, w, 3)
                outgoing_mask = create_outgoing_mask(F_list[i])
                diff = tf.abs(image_2_F - registrated_foreground)
                F_registrated.append(
                    tf.concat([image_2_F, image_2_B, key_frame, registrated_foreground, outgoing_mask, diff], -1))

            for i in range(len(I_list)):
                F_registrated[i] = self.sub_net(F_registrated[i])

            F_max = tf.reduce_max(tf.stack(F_registrated, 1), 1)
            F_mean = tf.reduce_mean(tf.stack(F_registrated, 1), 1)

            x = tf.concat([F_max, F_mean, image_2_F, image_2_B, key_frame], -1)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(3, (3, 3), (1, 1), 'same')(x)
            return image_2_F + x

    def FusionLayer_B(self, image_2_B, image_2_F, key_frame, I_list, F_list, lvl):
        with tf.variable_scope("FusionLayer_B_" + str(lvl), reuse=tf.AUTO_REUSE):
            b, h, w, _ = tf.unstack(tf.shape(image_2_B))

            B_registrated = []
            for i in range(len(I_list)):
                registrated_background = self.warp(I_list[i], F_list[i], b, h, w, 3)
                outgoing_mask = create_outgoing_mask(F_list[i])
                diff = tf.abs(image_2_B - registrated_background)
                B_registrated.append(
                    tf.concat([image_2_B, image_2_F, key_frame, registrated_background, outgoing_mask, diff], -1))

            for i in range(len(I_list)):
                B_registrated[i] = self.sub_net(B_registrated[i])

            B_max = tf.reduce_max(tf.stack(B_registrated, 1), 1)
            B_mean = tf.reduce_mean(tf.stack(B_registrated, 1), 1)

            x = tf.concat([B_max, B_mean, image_2_B, image_2_F, key_frame], -1)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(3, (3, 3), (1, 1), 'same')(x)
            return image_2_B + x

    def warp(self, I, F, b, h, w, c):
        return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [b, h, w, c])

    def _build_model(self, input_images,
                     F_last,
                     B_last,
                     FF,
                     FB):

        b = self.batch_size
        h = self.CROP_PATCH_H // (2 ** self.level)
        w = self.CROP_PATCH_W // (2 ** self.level)

        I = []
        for input_image in input_images:
            I.append(tf.image.resize_bilinear(input_image, (h, w), align_corners=True))

        if self.level == 4:
            F_last = []
            B_last = []
            for i in range(len(input_images)):
                tmp = []
                for j in range(len(input_images)):
                    tmp.append(self.warp(I[j], FF[i][j], b, h, w, 3))
                F_last.append(tf.reduce_mean(tf.stack(tmp, 1), 1))
                tmp = []
                for j in range(len(input_images)):
                    tmp.append(self.warp(I[j], FB[i][j], b, h, w, 3))
                B_last.append(tf.reduce_mean(tf.stack(tmp, 1), 1))

        else:
            for i in range(len(input_images)):
                for j in range(len(input_images)):
                    FF[i][j] = tf.image.resize_bilinear((FF[i][j] * 2.0), (h, w), align_corners=True)
                    FB[i][j] = tf.image.resize_bilinear((FB[i][j] * 2.0), (h, w), align_corners=True)
                F_last[i] = tf.image.resize_bilinear(F_last[i], (h, w), align_corners=True)
                B_last[i] = tf.image.resize_bilinear(B_last[i], (h, w), align_corners=True)

        F_pred = []
        B_pred = []
        for i in range(len(input_images)):
            F_pred.append(self.FusionLayer_F(F_last[i], B_last[i], I[i], I, FF[i], self.level))
            B_pred.append(self.FusionLayer_B(B_last[i], F_last[i], I[i], I, FB[i], self.level))

        return F_pred, B_pred

class ImageReconstruction_fence_arbitraryFrameNum_large_FBconcat_AvgMeanPool(object):
    def __init__(self, batch_size, CROP_PATCH_H, CROP_PATCH_W, level=4):
        self.batch_size = batch_size
        self.CROP_PATCH_H = CROP_PATCH_H
        self.CROP_PATCH_W = CROP_PATCH_W
        self.level = level

    def sub_net(self, x):
        with tf.variable_scope("sub_net", reuse=tf.AUTO_REUSE):
            x = tf.nn.elu(tf.layers.conv2d(x, 32, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            x = tf.nn.elu(tf.layers.conv2d(x, 64, 3, 1, 'same'))
            return x

    def FusionLayer_B(self, image_2_B, image_2_A, key_frame, I_list, F_list, lvl):
        with tf.variable_scope("FusionLayer_B_" + str(lvl), reuse=tf.AUTO_REUSE):
            b, h, w, _ = tf.unstack(tf.shape(image_2_B))

            B_registrated = []
            for i in range(len(I_list)):
                registrated_background = self.warp(I_list[i], F_list[i], b, h, w, 3)
                outgoing_mask = create_outgoing_mask(F_list[i])
                diff = tf.abs(image_2_B - registrated_background)
                B_registrated.append(
                    tf.concat([image_2_B, image_2_A, key_frame, registrated_background, outgoing_mask, diff], -1))

            for i in range(len(I_list)):
                B_registrated[i] = self.sub_net(B_registrated[i])

            B_max = tf.reduce_max(tf.stack(B_registrated, 1), 1)
            B_mean = tf.reduce_mean(tf.stack(B_registrated, 1), 1)

            x = tf.concat([B_max, B_mean, image_2_B, image_2_A, key_frame], -1)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(4, (3, 3), (1, 1), 'same')(x)
            return image_2_B + x[..., 0:3], image_2_A + x[..., 3:4]

    def warp(self, I, F, b, h, w, c):
        return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [b, h, w, c])

    def _build_model(self, input_images,
                     B_last,
                     A_last,
                     FB):

        b = self.batch_size
        h = self.CROP_PATCH_H // (2 ** self.level)
        w = self.CROP_PATCH_W // (2 ** self.level)

        I = []
        for input_image in input_images:
            I.append(tf.image.resize_bilinear(input_image, (h, w), align_corners=True))

        if self.level == 4:
            A_last = []
            B_last = []
            for i in range(len(input_images)):
                tmp = []
                for j in range(len(input_images)):
                    tmp.append(self.warp(I[j], FB[i][j], b, h, w, 3))
                B_last.append(tf.reduce_mean(tf.stack(tmp, 1), 1))
                A_last.append(tf.zeros([b, h, w, 1]))

        else:
            for i in range(len(input_images)):
                for j in range(len(input_images)):
                    FB[i][j] = tf.image.resize_bilinear((FB[i][j] * 2.0), (h, w), align_corners=True)
                B_last[i] = tf.image.resize_bilinear(B_last[i], (h, w), align_corners=True)
                A_last[i] = tf.image.resize_bilinear(A_last[i], (h, w), align_corners=True)

        A_pred = []
        B_pred = []
        for i in range(len(input_images)):
            tmp_B, tmp_A = self.FusionLayer_B(B_last[i], A_last[i], I[i], I, FB[i], self.level)
            B_pred.append(tmp_B)
            A_pred.append(tmp_A)

        return B_pred, A_pred

