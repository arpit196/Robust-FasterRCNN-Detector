#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/vgg16.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# TensorFlow/Keras implementation of the VGG-16 backbone for use as a feature
# extractor in Faster R-CNN. Only the convolutional layers are used.
#

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import layers

class LocalInstanceNormalizationL1(layers.Layer):
    def __init__(self, block_size=6, epsilon=1e-5, **kwargs):
        super(LocalInstanceNormalizationL1, self).__init__(**kwargs)
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        self.block_size = block_size
        self.epsilon = epsilon
        self.glo = layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        # input_shape will be (batch_size, height, width, channels)
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension must be known for LocalInstanceNormalizationL1.')
        self.gamma = self.add_weight(
            name='gamma',
            shape=(channels,), # One gamma value per channel
            initializer='ones', # Typically initialized to ones
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(channels,), # One beta value per channel
            initializer='zeros', # Typically initialized to zeros
            trainable=True
        )
        self.nweight = self.add_weight(
            name='nweight',
            shape=(channels,), # One beta value per channel
            initializer='zeros', # Typically initialized to zeros
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
        )
        
        super(LocalInstanceNormalizationL1, self).build(input_shape)

    def call(self, inputs):
        # Ensure inputs are float32 for consistent calculations
        inputs = tf.cast(inputs, tf.float32)

        # Reshape gamma and beta to (1, 1, 1, channels) for broadcasting across
        # batch, height, and width dimensions during the final scaling and shifting.
        gamma_reshaped = tf.reshape(self.gamma, [1, 1, 1, -1])
        beta_reshaped = tf.reshape(self.beta, [1, 1, 1, -1])
        local_mean = tf.nn.avg_pool(
            inputs,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )
        
        batch_mean, batch_sigma = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=True)
        batch_sigma = tf.reduce_mean(tf.abs(inputs-batch_mean),[0,1,2],keepdims=True)
        x_batch = (inputs - batch_mean) / (batch_sigma + self.epsilon)
        abs_diff = tf.abs(inputs - local_mean)

        local_mad = tf.nn.avg_pool(
            abs_diff,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )
        
        normalized_output = (inputs - local_mean) / (local_mad + self.epsilon)
        output = self.nweight*normalized_output + (1-self.nweight)*x_batch
        
        output = output * gamma_reshaped + beta_reshaped; #outputs_bn = self.bn(inputs)
        #output = self.nweight*output_inst + (1-self.nweight)*outputs_bn
        return output

class FeatureExtractor(tf.keras.Model):
  def __init__(self, l2 = 0):
    super().__init__()

    initial_weights = glorot_normal()
    regularizer = tf.keras.regularizers.l2(l2)
    input_shape = (None, None, 3)
  
    # First two convolutional blocks are frozen (not trainable)
    self._block1_conv1 = Conv2D(name = "block1_conv1", input_shape = input_shape, kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block1_in1 = BatchNormalization() # LocalInstanceNormalizationL1() #
    self._block1_conv2 = Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block1_in2 = BatchNormalization() # LocalInstanceNormalizationL1() #
    self._block1_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    self._block2_conv1 = Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block2_in1 = BatchNormalization() #LocalInstanceNormalizationL1() #
    self._block2_conv2 = Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, trainable = False)
    self._block2_in2 = BatchNormalization() #LocalInstanceNormalizationL1() #
    self._block2_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    # Weight decay begins from these layers onward: https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt
    self._block3_conv1 = Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block3_in1 = BatchNormalization() #LocalInstanceNormalizationL1() #
    self._block3_conv2 = Conv2D(name = "block3_conv2", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block3_in2 = BatchNormalization() #LocalInstanceNormalizationL1() #
    self._block3_conv3 = Conv2D(name = "block3_conv3", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    #self._block3_in3 = LocalInstanceNormalizationL1()
    self._block3_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    self._block4_conv1 = Conv2D(name = "block4_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    #self._block4_in1 = LocalInstanceNormalizationL1()
    self._block4_conv2 = Conv2D(name = "block4_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    #self._block4_in2 = LocalInstanceNormalizationL1()
    self._block4_conv3 = Conv2D(name = "block4_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    self._block4_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

    self._block5_conv1 = Conv2D(name = "block5_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    #self._block5_in1 = LocalInstanceNormalizationL1()
    self._block5_conv2 = Conv2D(name = "block5_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
    #self._block5_in2 = LocalInstanceNormalizationL1()
    self._block5_conv3 = Conv2D(name = "block5_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)

  def call(self, input_image):
    y = self._block1_conv1(input_image)
    y = self._block1_in1(y)
    y = self._block1_conv2(y)
    y = self._block1_in2(y)
    y = self._block1_maxpool(y)

    y = self._block2_conv1(y)
    y = self._block2_in1(y)
    y = self._block2_conv2(y)
    y = self._block2_in2(y)
    y = self._block2_maxpool(y)

    y = self._block3_conv1(y)
    y = self._block3_conv2(y)
    y = self._block3_conv3(y)
    y = self._block3_maxpool(y)

    y = self._block4_conv1(y)
    y = self._block4_conv2(y)
    y = self._block4_conv3(y)
    y = self._block4_maxpool(y)

    y = self._block5_conv1(y)
    y = self._block5_conv2(y)
    y = self._block5_conv3(y)

    return y
