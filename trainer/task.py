
import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

import argparse
import os

from .estimator import get_estimator
from .datasets import get_train_fn, get_eval_fn


def generate_experiment_fn(**experiment_args):
  """Create an experiment function.

  See command line help text for description of args.
  Args:
    experiment_args: keyword arguments to be passed through to experiment
      See `tf.contrib.learn.Experiment` for full args.
  Returns:
    A function:
      (tf.contrib.learn.RunConfig, tf.contrib.training.HParams) -> Experiment

    This function is used by learn_runner to create an Experiment which
    executes model code provided in the form of an Estimator and
    input functions.
  """

  def _experiment_fn(run_config, hparams):
    #train_files = ['/media/penn/DATA2/data/tfrecords/size_128/1.tfrecord']
    #eval_files = ['/media/penn/DATA2/data/tfrecords/size_128/0.tfrecord']
    #train_files = ['gs://tsa_pggan_data/size_128/{}.tfrecord' for i in range(1,12)]
    #eval_files = ['gs://tsa_pggan_data/size_128/eval/0.tfrecord']

    train_input_fn = get_train_fn(hparams.train_files, hparams.train_batch_size)
    eval_input_fn = get_eval_fn(hparams.eval_files, hparams.eval_batch_size)
    return tf.contrib.learn.Experiment(
            get_estimator(run_config, hparams),
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            **experiment_args)
  return _experiment_fn


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )

  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=8
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=8
  )
  parser.add_argument(
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True
  )
  # Training arguments
  parser.add_argument(
      '--learning-rate',
      help='Learning rate for the optimizer',
      default=1e-4,
      type=float
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      default='gs://tsa_pggan_data/logdir/run1',
      #required=True
  )
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
      help='Set logging verbosity'
  )
  # Experiment arguments
  parser.add_argument(
      '--eval-delay-secs',
      help='How long to wait before running first evaluation',
      default=1000,
      type=int
  )
  parser.add_argument(
      '--min-eval-frequency',
      help='Minimum number of training steps between evaluations',
      default=10000,
      type=int
  )
  parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help="""\
      Number of steps to run evalution for at each checkpoint.
      If unspecified will run until the input from --eval-files is exhausted
      """,
      default=None,
      type=int
  )
  """
  parser.add_argument(
      '--export-format',
      help='The input format of the exported SavedModel binary',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='JSON'
  )
  """

  args = parser.parse_args()

  print(args.eval_files)
  print(args.train_files)

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  # learn_runner pulls configuration information from environment
  # variables using tf.learn.RunConfig and uses this configuration
  # to conditionally execute Experiment, or param server code
  learn_runner.run(
      generate_experiment_fn(
          min_eval_frequency=args.min_eval_frequency,
          eval_delay_secs=args.eval_delay_secs,
          train_steps=args.train_steps,
          eval_steps=args.eval_steps,
      ),
      run_config=tf.contrib.learn.RunConfig(model_dir=args.job_dir),
      hparams=hparam.HParams(**args.__dict__)
  )
