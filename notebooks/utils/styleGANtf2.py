

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras import layers


# ------------------------------------------ Custom layers: ------------------------------------------

class PixelNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def call(self, inputs, epsilon=1e-8):
        epsilon = tf.constant(epsilon, dtype=inputs.dtype, name='epsilon')
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)

    def get_config(self):
        config = super(PixelNorm, self).get_config()
        return config


class Broadcast(layers.Layer):
    def __init__(self, dlatent_broadcast, **kwargs):
        super(Broadcast, self).__init__(**kwargs)
        self.dlatent_broadcast = dlatent_broadcast

    def call(self, inputs):
        return tf.tile(inputs[:, np.newaxis], [1, self.dlatent_broadcast, 1])

    def get_config(self):
        config = super(Broadcast, self).get_config()
        config['dlatent_broadcast'] = self.dlatent_broadcast
        return config


class Truncation(layers.Layer):
    def __init__(self, num_layers, psi, cutoff, x_avg_shape, **kwargs):
        super(Truncation, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.psi = psi
        self.cutoff = cutoff
        self.x_avg_shape = x_avg_shape
        self.x_avg = tf.Variable(np.zeros(self.x_avg_shape).astype(np.float32), trainable=False)

    def call(self, x):
        layer_idx = np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
        ones = np.ones(layer_idx.shape, dtype=np.float32)
        coefs = tf.where(layer_idx < self.cutoff, self.psi * ones, ones)

        return self.x_avg + (x - self.x_avg) * coefs

    def get_config(self):
        config = super(Truncation, self).get_config()
        config['num_layers'] = self.num_layers
        config['psi'] = self.psi
        config['cutoff'] = self.cutoff
        config['x_avg_shape'] = self.x_avg_shape
        return config


class Slice(layers.Layer):
    def __init__(self, layer_idx, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.layer_idx = layer_idx

    def call(self, x):
        return x[:, self.layer_idx]

    def get_config(self):
        config = super(Slice, self).get_config()
        config['layer_idx'] = self.layer_idx
        return config


class StyleMod(layers.Layer):
    def __init__(self, data_format='channels_last', **kwargs):
        super(StyleMod, self).__init__(**kwargs)
        self.data_format = data_format

    def call(self, x, x_dlatents):
        if self.data_format == 'channels_last':
            x_dlatents = tf.reshape(x_dlatents, [-1, 2] + [1] * (len(x.shape) - 2) + [x.shape[-1]])
        else:
            x_dlatents = tf.reshape(x_dlatents, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (x_dlatents[:, 0] + 1) + x_dlatents[:, 1]

    def get_config(self):
        config = super(StyleMod, self).get_config()
        config['data_format'] = self.data_format
        return config


class InstanceNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        # self.x_shape = x_shape
        # self.x = tf.Variable(np.zeros(self.x_shape).astype(np.float32), trainable=False)

    def build(self, input_shapes):
        epsilon = 1e-8
        self.epsilon = tf.constant(epsilon, dtype=tf.float32, name='epsilon')
        super(InstanceNorm, self).build(input_shapes)

    def call(self, x):
        x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # channels_last
        x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + self.epsilon)  # channels_last
        return x


class BranchOut(layers.Layer):
    def __init__(self, kernel_dim, tile=False, **kwargs):
        self.kernel_dim = kernel_dim
        self.tile = tile
        super(BranchOut, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.kernel = self.add_weight(name='kernel', shape=self.kernel_dim, initializer='uniform', trainable=False)
        super(BranchOut, self).build(input_shapes)

    def call(self, x):
        if self.tile:
            return tf.tile(self.kernel, [tf.shape(x)[0], 1, 1, 1])
        return self.kernel

    def get_config(self):
        config = super(BranchOut, self).get_config()
        config['kernel_dim'] = self.kernel_dim
        config['tile'] = self.tile
        return config


class ApplyBias(layers.Layer):
    def __init__(self, bias_dim, data_format='channels_last', **kwargs):
        self.bias_dim = bias_dim
        self.data_format = data_format
        super(ApplyBias, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.bias = self.add_weight(name='bias', shape=self.bias_dim, initializer='uniform', trainable=False)
        super(ApplyBias, self).build(input_shapes)

    def call(self, x):
        if len(x.shape) == 2:
            return x + self.bias
        if self.data_format == 'channels_last':
            return x + tf.reshape(self.bias, [1, 1, 1, -1])
        return x + self.bias #tf.reshape(self.bias, [1, -1, 1, 1])

    def get_config(self):
        config = super(ApplyBias, self).get_config()
        config['bias_dim'] = self.bias_dim
        return config


class ApplyNoise(layers.Layer):
    def __init__(self, noise_dim, weight_dim, data_format='channels_last', **kwargs):
        self.noise_dim = noise_dim
        self.weight_dim = weight_dim
        self.data_format = data_format
        super(ApplyNoise, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.noise = self.add_weight(name='noise', shape=self.noise_dim, initializer='uniform', trainable=False)
        self.weight = self.add_weight(name='noise', shape=self.weight_dim, initializer='uniform', trainable=False)
        super(ApplyNoise, self).build(input_shapes)

    def call(self, x):
        if self.data_format == 'channels_last':
            return x + self.noise * tf.reshape(tf.cast(self.weight, x.dtype), [1, 1, 1, -1])
            # return x + self.noise * tf.cast(self.weight, x.dtype)
        return x + self.noise * tf.reshape(tf.cast(self.weight, x.dtype), [1, -1, 1, 1])

    def get_config(self):
        config = super(ApplyNoise, self).get_config()
        config['noise_dim'] = self.noise_dim
        config['weight_dim'] = self.weight_dim
        config['data_format'] = self.data_format
        return config


class Blur2d(layers.Layer):
    def __init__(self, f=[1, 2, 1], **kwargs):
        self.f = f
        super(Blur2d, self).__init__(**kwargs)

    def call(self, x):
        f = np.array(self.f, dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        f = f[::-1, ::-1]
        f = f[:, :, np.newaxis, np.newaxis]
        f = np.tile(f, [1, 1, int(x.shape[-1]), 1])  # channels_last

        f = tf.constant(f, dtype=x.dtype, name='filter')
        strides = [1, 1, 1, 1]
        x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NHWC')
        return x

    def get_config(self):
        config = super(Blur2d, self).get_config()
        config['f'] = self.f
        return config


class Upscale2d(layers.Layer):
    def __init__(self, factor=2, data_format='channels_last', **kwargs):
        self.factor = factor
        self.data_format = data_format
        super(Upscale2d, self).__init__(**kwargs)

    def call(self, x):
        s = x.shape
        if self.data_format == 'channels_last':
            x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
            x = tf.tile(x, [1, 1, self.factor, 1, self.factor, 1])
            x = tf.reshape(x, [-1, s[1] * self.factor, s[2] * self.factor, s[3]])
        else:
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, self.factor, 1, self.factor])
            x = tf.reshape(x, [-1, s[1], s[2] * self.factor, s[3] * self.factor])
        return x

    def get_config(self):
        config = super(Upscale2d, self).get_config()
        config['factor'] = self.factor
        config['data_format'] = self.data_format
        return config


class LerpClip(layers.Layer):
    def __init__(self, lod=0.0, **kwargs):
        self.lod = lod
        super(LerpClip, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.lod_tf = tf.cast(self.lod, tf.float32)
        super(LerpClip, self).build(input_shapes)

    def call(self, a, b):
        return a + (b - a) * tf.clip_by_value(self.lod_tf, 0.0, 1.0)

    def get_config(self):
        config = super(LerpClip, self).get_config()
        config['lod'] = self.lod
        return config
