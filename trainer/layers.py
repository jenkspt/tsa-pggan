import tensorflow as tf
import numpy as np
from functools import partial

def fromVolume(inputs, filters, name='FromVolume', reuse=tf.AUTO_REUSE):
    return conv3d(inputs, filters, kernel_size=1, name=name, reuse=reuse)

def toVolume(inputs, channels=1, name='ToVolume', reuse=tf.AUTO_REUSE):
    return conv3d(inputs, filters=channels, kernel_size=1, name=name, reuse=reuse)

def nf(stage, fmap_base=512, fmap_max = 512, fmap_decay=1):
    """ Calculates the number of filters for the current stage """
    num_filters = int(fmap_base / (2.0 ** (stage * fmap_decay)))
    return min(num_filters, fmap_max)

def select_output_layer(output_layers, current_res, alpha, name='SelectOutputLayer'):
    with tf.name_scope('if_fade'):
        is_fade = (alpha > 0) & (alpha < 1)
    
    pred_list = [(tf.equal(current_res, 4), lambda i=0:output_layers[i])]
    for i in range(1, len(output_layers)):
        res = 2**(i+2) 
        pred_list += [(tf.equal(current_res, res), 
            lambda j=i: tf.cond(is_fade, 
                lambda k=j: (output_layers[k] * alpha) + (output_layers[k-1] * (1-alpha)),
                lambda k=j: output_layers[k]))]

    return tf.case(pred_list, default=lambda i=-1: output_layers[i], exclusive=True)

def select_input_layer(layer, inputs,
    res, current_res, alpha, name='SelectInputLayer'):

    assert inputs.get_shape().as_list() == layer.get_shape().as_list()


    with tf.variable_scope('SelectInputTo{0}x{0}'.format(res)):
        # Training resolution is higher than the current block
        with tf.name_scope('if_higher_res'):
            is_higher_res = (current_res > res)
        # Training resolution is equal to the current block
        with tf.name_scope('if_current_res'):
            is_current_res = tf.equal(current_res, res)
        # Transitioning to the next block
        with tf.name_scope('if_fade'):
            is_fade = (alpha > 0) & (alpha < 1)


        def apply_fade():
            with tf.name_scope('fade_layer'):
                return (inputs * (1-alpha)) + (layer * alpha)
        def get_inputs(): return inputs
        def get_layer(): return layer

        output_layer = tf.cond(is_higher_res, get_layer,
                lambda: tf.cond(is_current_res & is_fade, apply_fade,
                    get_inputs))

    return output_layer

def minibatch_std(batch, layer, is_training=True):
    with tf.name_scope('minibatch_stddev'):
        if is_training:
            # Compute std for each pixel location
            mean, var = tf.nn.moments(batch, axes=0)
            # mean & var have shape [height, width, depth, channels]
            avg_std = tf.reduce_mean(tf.sqrt(var), axis=[0, 1, 2])[0]
            streaming_avg_std, _ = tf.metrics.mean(avg_std)
            tf.summary.scalar('streaming_avg_std', _)
        else:
            # running out of time creating this model - so for predictions,
            # avg_std is set manually based on the streaming mean from training
            avg_std = tf.constant(.2)
            
        layer_shape = tf.shape(layer)
        shape = [layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3], 1]

        filled = tf.fill(shape, avg_std)
        return tf.concat([layer, filled], axis=-1)
        

def pixelwise_norm(layer, epsilon=1e-8):
    """
    Progressive Growing of Gans 4.2
    Args:
        layer (Tensor): with shape [batch_size, width, height, channel]
        epsilon  (int): Not sure ... assuming it prevents zero values
    """
    with tf.name_scope('pixelwise_norm'):
        return layer / tf.sqrt(tf.reduce_mean(layer**2, axis=-1, keep_dims=True) + epsilon)

def weight_initializer(he_init=True, stride=1, transpose=False):

    """Returns an initializer that generates tensors without scaling variance.
    Args:
    Returns:
    An initializer that generates tensors with unit variance.
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating point type.')

        # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
        # This is the right thing for matrix multiply and convolutions.
        shape = list(map(float, shape))
        if shape:
            fan_in = shape[-2] if len(shape) > 1 else shape[-1]
            fan_out = shape[-1]
        else:
            fan_in = 1.0
            fan_out = 1.0

        for dim in shape[:-2]:
            fan_in *= dim
            fan_out *= dim

        if transpose:
            fan_in /= stride**len(shape[:-1])
        else:
            fan_out /= stride**len(shape[:-1])

        if he_init:
            stddev = np.sqrt(4./(fan_in+fan_out))
        else:
            stddev = np.sqrt(2./(fan_in+fan_out))

        shape = list(map(int, shape))
        return tf.random_normal(shape, stddev=stddev, dtype=dtype)

    return _initializer


# Set default values for layers
conv2d = partial(tf.layers.conv2d,
        kernel_size=3,
        padding='same',
        activation=tf.nn.leaky_relu,
        use_bias=True,
        kernel_initializer=weight_initializer(he_init=True))

conv3d = partial(tf.layers.conv3d,
        kernel_size=3,
        padding='same',
        activation=tf.nn.leaky_relu,
        use_bias=True)

dense = partial(tf.layers.dense, 
        activation=tf.nn.leaky_relu, 
        use_bias=True,
        kernel_initializer=weight_initializer(he_init=False))


def global_avg_pool(input, name='global_avg_pool'):
    """
    Averages each channel to a single value

    Args:
        input (Tensor): 4d input tensor
    Returns:
        Tensor: With shape [batch_size, 1, 1, n_channels]
    """
    shape = input.get_shape().as_list()
    pool = tf.nn.pool(
            input, window_shape=shape[1:-1], 
            pooling_type='AVG', padding='VALID', name=name)
    return pool

def upsample3d(inputs, size=(2,2,2), name='UpSample3D'):
    """ Upsamples the layer by factors in size tuple """
    layer = tf.keras.layers.UpSampling3D(size, name=name)
    return layer.apply(inputs)

def downsample3d(inputs, size=(2,2,2), name='DownSample3D'):
    return tf.layers.average_pooling3d(inputs, pool_size=size, strides=size, name=name)
