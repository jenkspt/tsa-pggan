import tensorflow as tf

from . import model


def model_fn(features, labels, mode, params):
    """Model function used in the estimator.

    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.

    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    pggan = model.PGGAN(
            learning_rate=params.learning_rate)

    estimator_spec = pggan.get_estimator_spec(features, labels, mode)
    return estimator_spec

def get_estimator(run_config, params):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config)

