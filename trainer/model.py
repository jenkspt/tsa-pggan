import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.python.training import training_util

import six
from functools import partial
from .layers import *
from .hooks import *
from .losses import *

class PGGAN():

    def __init__(self,
            learning_rate=1e-4,
            beta1=0,
            beta2=.99,
            nclasses=17,
            nlatent=256,
            resolution=128, 
            start_res=4,
            stablize_steps=60000, 
            fade_steps=60000,
            G_scope='Generator',
            D_scope='Discriminator'):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.nclasses = nclasses
        self.nlatent = nlatent
        self.resolution = resolution
        self.G_scope = G_scope
        self.D_scope = D_scope

        self.R = int(np.log2(resolution))
        assert 2**self.R == self.resolution

        self.global_step = training_util.get_or_create_global_step()
        self.global_step_inc = self.global_step.assign_add(1)

        with tf.variable_scope('Scheduling'):
            self.res = tf.Variable(start_res, trainable=False, name='Resolution')
            self.alpha = tf.Variable(-1.0, trainable=False, name='Alpha')

            self.stablize_steps = tf.constant(stablize_steps, dtype=tf.float32)
            self.fade_steps = tf.constant(fade_steps, dtype=tf.float32)

            self.stablize_increment = \
                    tf.assign(self.alpha, self.alpha + 1.0/self.stablize_steps)
            self.fade_increment = \
                    tf.assign(self.alpha, self.alpha + 1.0/self.fade_steps)

            self.res_increment = tf.assign(self.res, self.res*2)
            self.reset_alpha = tf.assign(self.alpha, -1.0)

    def get_estimator_spec(self, real_features, real_class_labels, mode):

        with tf.variable_scope(self.D_scope) as dscope:
            is_training = True if mode == 'train' or mode == 'infer' else False
            real_score, real_logits = self.discriminator(real_features, is_training)
            tf.summary.scalar('accuracy', accuracy(real_class_labels, real_logits))

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=real_logits)

        with tf.variable_scope('Latent'):
            noise = tf.random_normal(
                    [tf.shape(real_class_labels)[0], self.nlatent], 
                    dtype=tf.float32,
                    name='Z')
            
            with tf.name_scope('fake_class_labels'):
                shape = tf.shape(real_class_labels)
                random = tf.random_uniform(shape=[shape[0], shape[1]], maxval=1)
                fake_class_labels = tf.cast(random < 0.096, dtype=tf.float32)


        with tf.variable_scope(self.G_scope):
            fake_features = self.generator(noise, fake_class_labels)

        with tf.variable_scope(self.D_scope, reuse=True):
            fake_score, fake_logits = self.discriminator(fake_features, is_training)
        
        with ops.name_scope('losses'):
            loss_tuple = gan_loss(
                discriminator_fn = self.discriminator,
                discriminator_scope = self.D_scope,
                real_features = real_features,
                fake_features = fake_features,
                disc_real_score = real_score,
                disc_fake_score = fake_score,
                disc_real_logits = real_logits,
                disc_fake_logits = fake_logits,
                real_class_labels = real_class_labels,
                fake_class_labels = fake_class_labels)
           
            total_loss = loss_tuple.discriminator_loss + loss_tuple.generator_loss

        generator_variables = variables_lib.get_trainable_variables(self.G_scope)
        discriminator_variables = variables_lib.get_trainable_variables(self.D_scope)

        G_train_op = tf.train.AdamOptimizer(
                self.learning_rate, 
                self.beta1, 
                self.beta2, 
                name='generator_optimizer').minimize(
                        loss_tuple.generator_loss, 
                        var_list=generator_variables)

        D_train_op = tf.train.AdamOptimizer(
                self.learning_rate, 
                self.beta1, 
                self.beta2,
                name='discriminator_optimizer').minimize(\
                        loss_tuple.discriminator_loss, 
                        var_list=discriminator_variables)
        
        train_hook = PGTrainHook(
                G_train_op, 
                D_train_op, 
                self.alpha, 
                self.res,
                self.stablize_increment,
                self.fade_increment,
                self.res_increment,
                self.reset_alpha)
        eval_metric_ops = get_eval_metric_ops(real_class_labels, real_logits)
        
        return tf.estimator.EstimatorSpec(
                loss=total_loss,
                mode=mode,
                train_op=self.global_step_inc,
                training_hooks = [train_hook],
                eval_metric_ops=None)
        """
        needs to return loss, train_op, train_hook
        """

    def generator(self, noise, labels=None):
        """
        Args:
        latent (Tensor):
        """ 
        latent = tf.concat([noise, labels], axis=-1)
        layer = pixelwise_norm(latent)
        output_layers = []
        with tf.variable_scope('{0}x{0}x{0}'.format(4)):
            # Shape is [batch, height, width, depth, channels]
            layer = tf.reshape(layer, [-1, 1, 1, 1, self.nclasses + self.nlatent])
            layer = tf.pad(layer, [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")
            layer = conv3d(layer, nf(1), kernel_size=4, padding='valid')
            layer = pixelwise_norm(layer)
            layer = conv3d(layer, nf(1))
            layer = pixelwise_norm(layer)
            print('G4x4x4', layer.get_shape())
        with tf.variable_scope('ReshapeOutputFrom{0}x{0}x{0}'.format(4)):
            out = toVolume(layer)
            #out = tf.image.resize_images(out, [2**self.R, 2**self.R], method=1)
            out = upsample3d(out, self.resolution//4)
        output_layers = [out]
        
        for S in range(2, self.R):
            #layer = tf.image.resize_images(layer, [2**(S+1), 2**(S+1)], method=1)
            layer = upsample3d(layer, 2, name='Upsample_x2')    
            with tf.variable_scope('{0}x{0}x{0}'.format(2**(S+1))):
                layer = conv3d(layer, nf(S), name='conv3d_a')
                layer = pixelwise_norm(layer)
                layer = conv3d(layer, nf(S), name='conv3d_b')
                layer = pixelwise_norm(layer)
                print('G{0}x{0}x{0} '.format(2**(S+1)), layer.get_shape())

            with tf.variable_scope('ReshapeOutputFrom{0}x{0}x{0}'.format(2**(S+1))):
                out = toVolume(layer)
                #out = tf.image.resize_images(out, [2**self.R, 2**self.R], method=1)
                out = upsample3d(out, self.resolution//(2**(S+1)), name='UpsampleFullRes')
            output_layers.append(out)
        # Control-flow class depends on stage and alpha
        output = select_output_layer(output_layers, self.res, self.alpha)
        tf.summary.image('front_view', tf.reduce_mean(output, axis=1))
        tf.summary.image('side_view', tf.reduce_mean(output, axis=2))
        tf.summary.image('top_view', tf.reduce_mean(output, axis=3))
        return output




    def discriminator(self, inputs, is_training=True):

        layer = fromVolume(inputs, nf(self.R-1))
        for S in range(self.R-1, 1, -1): # S = {4, 3, 2}
            # convolve
            with tf.variable_scope('{0}x{0}x{0}'.format(2**(S+1))):
                layer = conv3d(layer, nf(S), name='conv3d_a')
                layer = conv3d(layer, nf(S-1), name='conv3d_b')
            # Pool
            name = 'PoolTo{0}x{0}x{0}'.format(2**(S))
            layer = downsample3d(layer, 2, name=name)
            # Resize and add feature maps to input
            with tf.variable_scope('ReshapeInputTo{0}x{0}x{0}'.format(2**(S))):
                in_volume = downsample3d(inputs, 2**(self.R-S), name='Resize')
                in_layer = fromVolume(in_volume, nf(S-1))
            # Graph logic for selecting input resolution
            layer = select_input_layer(layer, in_layer, 2**S, self.res, self.alpha)

        with tf.variable_scope('{0}x{0}x{0}'.format(4)):
            layer = minibatch_std(inputs, layer, is_training) # +1 channels
            layer = conv3d(layer, nf(1), name='conv3d_a')
            layer = conv3d(layer, nf(0), kernel_size=4, padding='valid', name='conv3d_b')
        classifier = dense(layer, units=1+self.nclasses, name='Classifier')
        classifier = tf.reshape(classifier, [-1, 1+self.nclasses])
        score, class_logits = classifier[:,:1], classifier[:,1:]
        return score, class_logits

def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.

    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy')
    }

def accuracy(labels, predictions):
    labels = tf.cast(labels, tf.bool)
    predictions = (predictions > 0.5)
    return tf.reduce_mean(tf.cast((predictions == labels), tf.float32))
