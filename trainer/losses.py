from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.framework.python.ops import variables as contrib_variables_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import distribution as ds
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.summary import summary

from collections import namedtuple

def wasserstein_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Wasserstein generator loss for GANs.
  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add detailed summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'generator_wasserstein_loss', (
      discriminator_gen_outputs, weights)) as scope:
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)

    loss = - discriminator_gen_outputs
    loss = losses.compute_weighted_loss(
        loss, weights, scope, loss_collection, reduction)

    if add_summaries:
      summary.scalar('generator_wass_loss', loss)

  return loss

def wasserstein_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Wasserstein discriminator loss for GANs.
  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'discriminator_wasserstein_loss', (
      discriminator_real_outputs, discriminator_gen_outputs, real_weights,
      generated_weights)) as scope:
    discriminator_real_outputs = math_ops.to_float(discriminator_real_outputs)
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
    discriminator_real_outputs.shape.assert_is_compatible_with(
        discriminator_gen_outputs.shape)

    loss_on_generated = losses.compute_weighted_loss(
        discriminator_gen_outputs, generated_weights, scope,
        loss_collection=None, reduction=reduction)
    loss_on_real = losses.compute_weighted_loss(
        discriminator_real_outputs, real_weights, scope, loss_collection=None,
        reduction=reduction)
    loss = loss_on_generated - loss_on_real
    util.add_loss(loss, loss_collection)

    if add_summaries:
      summary.scalar('discriminator_gen_wass_loss', loss_on_generated)
      summary.scalar('discriminator_real_wass_loss', loss_on_real)
      summary.scalar('discriminator_wass_loss', loss)

  return loss

def wasserstein_gradient_penalty(
    real_data,
    generated_data,
    generator_inputs,
    discriminator_fn,
    discriminator_scope,
    epsilon=1e-10,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """The gradient penalty for the Wasserstein discriminator loss.
  See `Improved Training of Wasserstein GANs`
  (https://arxiv.org/abs/1704.00028) for more details.
  Args:
    real_data: Real data.
    generated_data: Output of the generator.
    generator_inputs: Exact argument to pass to the generator, which is used
      as optional conditioning to the discriminator.
    discriminator_fn: A discriminator function that conforms to TFGAN API.
    discriminator_scope: If not `None`, reuse discriminators from this scope.
    epsilon: A small positive number added for numerical stability when
      computing the gradient norm.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data` and `generated_data`, and must be broadcastable to
      them (i.e., all dimensions must be either `1`, or the same as the
      corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  Raises:
    ValueError: If the rank of data Tensors is unknown.
  """
  real_data = ops.convert_to_tensor(real_data)
  generated_data = ops.convert_to_tensor(generated_data)
  if real_data.shape.ndims is None:
    raise ValueError('`real_data` can\'t have unknown rank.')
  if generated_data.shape.ndims is None:
    raise ValueError('`generated_data` can\'t have unknown rank.')

  differences = generated_data - real_data
  batch_size = differences.shape[0].value or array_ops.shape(differences)[0]
  alpha_shape = [batch_size] + [1] * (differences.shape.ndims - 1)
  alpha = random_ops.random_uniform(shape=alpha_shape)
  interpolates = real_data + (alpha * differences)

  # Reuse variables if a discriminator scope already exists.
  reuse = False if discriminator_scope is None else True
  with variable_scope.variable_scope(discriminator_scope, 'gpenalty_dscope',
                                     reuse=reuse):
    disc_interpolates = discriminator_fn(interpolates, generator_inputs)

  if isinstance(disc_interpolates, tuple):
    # ACGAN case: disc outputs more than one tensor
    disc_interpolates = disc_interpolates[0]

  gradients = gradients_impl.gradients(disc_interpolates, interpolates)[0]
  gradient_squares = math_ops.reduce_sum(
      math_ops.square(gradients), axis=list(range(1, gradients.shape.ndims)))
  # Propagate shape information, if possible.
  if isinstance(batch_size, int):
    gradient_squares.set_shape([
        batch_size] + gradient_squares.shape.as_list()[1:])
  # For numerical stability, add epsilon to the sum before taking the square
  # root. Note tf.norm does not add epsilon.
  slopes = math_ops.sqrt(gradient_squares + epsilon)
  penalties = math_ops.square(slopes - 1.0)
  penalty = losses.compute_weighted_loss(
      penalties, weights, scope=scope, loss_collection=loss_collection,
      reduction=reduction)

  if add_summaries:
    summary.scalar('gradient_penalty_loss', penalty)

  return penalty

def acgan_generator_loss(
    discriminator_gen_classification_logits,
    fake_class_labels,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """ACGAN loss for the generator.
  The ACGAN loss adds a classification loss to the conditional discriminator.
  Therefore, the discriminator must output a tuple consisting of
    (1) the real/fake prediction and
    (2) the logits for the classification (usually the last conv layer,
        flattened).
  For more details:
    ACGAN: https://arxiv.org/abs/1610.09585
  Args:
    discriminator_gen_classification_logits: Classification logits for generated
      data.
    fake_class_labels,
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_classification_logits`, and must be broadcastable to
      `discriminator_gen_classification_logits` (i.e., all dimensions must be
      either `1`, or the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. Shape depends on `reduction`.
  Raises:
    ValueError: if arg module not either `generator` or `discriminator`
    TypeError: if the discriminator does not output a tuple.
  """
  loss = losses.sigmoid_cross_entropy(
      fake_class_labels, discriminator_gen_classification_logits, weights=weights,
      scope=scope, loss_collection=loss_collection, reduction=reduction)

  if add_summaries:
    summary.scalar('generator_ac_loss', loss)

  return loss

# ACGAN losses from `Conditional Image Synthesis With Auxiliary Classifier GANs`
# (https://arxiv.org/abs/1610.09585).
def acgan_discriminator_loss(
    discriminator_real_classification_logits,
    discriminator_gen_classification_logits,
    real_class_labels,
    fake_class_labels,
    label_smoothing=0.0,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """ACGAN loss for the discriminator.
  The ACGAN loss adds a classification loss to the conditional discriminator.
  Therefore, the discriminator must output a tuple consisting of
    (1) the real/fake prediction and
    (2) the logits for the classification (usually the last conv layer,
        flattened).
  For more details:
    ACGAN: https://arxiv.org/abs/1610.09585
  Args:
    discriminator_real_classification_logits: Classification logits for real
      data.
    discriminator_gen_classification_logits: Classification logits for generated
      data.
    real_class_labels,
    fake_class_labels,
    label_smoothing: A float in [0, 1]. If greater than 0, smooth the labels for
      "discriminator on real data" as suggested in
      https://arxiv.org/pdf/1701.00160
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_classification_logits`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. Shape depends on `reduction`.
  Raises:
    TypeError: If the discriminator does not output a tuple.
  """
  loss_on_generated = losses.sigmoid_cross_entropy(
      fake_class_labels, discriminator_gen_classification_logits,
      weights=generated_weights, scope=scope, loss_collection=None,
      reduction=reduction)
  loss_on_real = losses.sigmoid_cross_entropy(
      real_class_labels, discriminator_real_classification_logits,
      weights=real_weights, label_smoothing=label_smoothing, scope=scope,
      loss_collection=None, reduction=reduction)
  loss = loss_on_generated + loss_on_real
  util.add_loss(loss, loss_collection)

  if add_summaries:
    summary.scalar('discriminator_gen_ac_loss', loss_on_generated)
    summary.scalar('discriminator_real_ac_loss', loss_on_real)
    summary.scalar('discriminator_ac_loss', loss)

  return loss


GANLoss = namedtuple('GANLoss', ['generator_loss', 'discriminator_loss'])
def gan_loss(
        discriminator_fn,
        discriminator_scope,
        real_features,
        fake_features,
        disc_real_score,
        disc_fake_score,
        disc_real_logits,
        disc_fake_logits,
        real_class_labels,
        fake_class_labels,
        gradient_penalty_weight=10,
        gradient_penalty_epsilon=1e-10,
        aux_cond_generator_weight=1,
        aux_cond_discriminator_weight=1,
        add_summaries=True):
    """Returns losses necessary to train generator and discriminator.
    Args:
        real_features:
        fake_features:
        real_logits:
        fake_logits:
        gradient_penalty_weight: If not `None`, must be a non-negative Python number 
            or Tensor indicating how much to weight the gradient penalty. See
            https://arxiv.org/pdf/1704.00028.pdf for more details.
        gradient_penalty_epsilon: If `gradient_penalty_weight` is not None, the
            small positive value used by the gradient penalty function for numerical
            stability. Note some applications will need to increase this value to
            avoid NaNs.
        mutual_information_penalty_weight: If not `None`, must be a non-negative
            Python number or Tensor indicating how much to weight the mutual
            information penalty. See https://arxiv.org/abs/1606.03657 for more
            details.
        aux_cond_generator_weight: If not None: add a classification loss as in
            https://arxiv.org/abs/1610.09585
        aux_cond_discriminator_weight: If not None: add a classification loss as in
            https://arxiv.org/abs/1610.09585
        add_summaries: Whether or not to add summaries for the losses.
    Returns:
        A GANLoss 2-tuple of (generator_loss, discriminator_loss). Includes
        regularization losses.
    """
    with ops.name_scope('Wasserstein_losses'):
        with ops.name_scope('generator_loss'):
            gen_loss = wasserstein_generator_loss(
                    disc_fake_score, add_summaries=add_summaries)
        with ops.name_scope('discriminator_loss'):
            disc_loss = wasserstein_discriminator_loss(
                    disc_real_score, disc_fake_score, add_summaries=add_summaries)
        
        with ops.name_scope('gradient_penalty'):
            gp_loss = wasserstein_gradient_penalty(
                    real_data=real_features,
                    generated_data=fake_features,
                    generator_inputs=None,
                    discriminator_fn=discriminator_fn,
                    discriminator_scope=discriminator_scope,
                    epsilon=gradient_penalty_epsilon,
                    weights=gradient_penalty_weight,
                    add_summaries=add_summaries)
    
    with ops.name_scope('Auxiliary_losses'):
        with ops.name_scope('generator_loss'):
            ac_gen_loss = acgan_generator_loss(
                    disc_fake_logits,
                    fake_class_labels,
                    weights=aux_cond_generator_weight,
                    add_summaries=add_summaries)

        with ops.name_scope('discriminator_loss'):
            ac_disc_loss = acgan_discriminator_loss(
                    disc_real_logits,
                    disc_fake_logits,
                    real_class_labels,
                    fake_class_labels,
                    real_weights=aux_cond_discriminator_weight,
                    add_summaries=add_summaries)

    return GANLoss(gen_loss + ac_gen_loss, disc_loss + gp_loss + ac_disc_loss)
