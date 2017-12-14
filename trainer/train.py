import tensorflow as tf

import estimator
from mnist import MNIST
import hooks
import model

tf.logging.set_verbosity(tf.logging.DEBUG)

BATCH_SIZE=128

params = tf.contrib.training.HParams(
        learning_rate=1e-4,
        beta1 = 0.5,
        beta2 = 0.9,
        stablize_steps=600.0,
        fade_steps = 600.0,
        train_steps = 2000)


train_input_fn, iterator_init_hook = get_train_inputs(BATCH_SIZE)

pggan = tf.estimator.Estimator(\
        model_fn=estimator.model_fn, params=params, model_dir='logdir/run2')
pggan.train(train_input_fn, hooks=[iterator_init_hook], steps=10000)
