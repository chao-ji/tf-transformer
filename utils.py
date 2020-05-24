"""Defines of utility functions."""
import math

import numpy as np
import tensorflow as tf


def get_padding_mask(inputs, padding_value=0):
  """Creates a tensor used to mask out padded tokens.

  Args:
    inputs: int tensor of shape [batch_size, src_seq_len], token ids
      of source sequences.
    padding_value: int scalar, the id corresponding to padded tokens in the 
      input. 

  Returns:
    mask: float tensor of shape [batch_size, 1, 1, src_seq_len], holding ones
      and zeros for regular token ids and padded token ids. 
  """
  mask = tf.cast(tf.equal(inputs, padding_value), 'float32') 
  mask = mask[:, tf.newaxis, tf.newaxis, :]
  return mask
 
 
def get_look_ahead_mask(seq_len):
  """Creates a tensor used to mask out future tokens in the target sequences at
  training time.

  Given sequence length `L` of target sequence, the mask would be a L x L
  matrix (when `tf.squeeze`ed) where upper diagonal entries are ones and all 
  other entries zeros.

  0, 1, 1, ..., 1
  0, 0, 1, ..., 1
  
      ... ...
  
  0, 0, 0, ..., 0

  Args:
    seq_len: int scalar tensor, sequence length.

  Returns:
    mask: float tensor of shape [1, 1, seq_len, seq_len], the mask tensor.
  """
  mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
  mask = mask[tf.newaxis, tf.newaxis, :, :]
  return mask


def get_positional_encoding(seq_len, hidden_size):
  """Creates a tensor that encodes positional information.

  Args:
    seq_len: int scalar tensor, sequence length.
    hidden_size: int scalar, the hidden size of continuous representation. 

  Returns:
    positional_encoding: float tensor of shape [seq_len, hidden_size], the 
      tensor that encodes positional information.
  """
  positions = tf.cast(tf.range(seq_len), 'float32')
  hidden_size //= 2
  log_increment = math.log(10000.) / (tf.cast(hidden_size, 'float32') - 1)
  depths = tf.exp(tf.cast(tf.range(hidden_size), 'float32') * -log_increment)
  
  positional_encoding = tf.expand_dims(positions, 1) * tf.expand_dims(depths, 0)
  positional_encoding = tf.concat([tf.sin(positional_encoding), 
                                   tf.cos(positional_encoding)], axis=1)
  return positional_encoding


def compute_loss(labels, logits, smoothing, vocab_size):
  """Computes average (per-token) cross entropy loss.

  Args:
    labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
      token ids.
    logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
      predicted logits of tokens over the vocabulary.
    smoothing: float scalar, the amount of label smoothing applied to the
      one-hot class labels. 
    vocab_size: int scalar, num of tokens (including SOS and EOS) in the 
      vocabulary.

  Returns:
    loss: float scalar tensor, the per-token cross entropy
  """
  effective_vocab_size = vocab_size - 1
  conf = 1.0 - smoothing    # 0.9
  low_conf = smoothing / (effective_vocab_size - 1) # close to 0.0

  # [batch_size, tgt_seq_len, vocab_size] 
  labels_one_hot = tf.one_hot(
      labels,
      depth=vocab_size,
      on_value=conf,
      off_value=low_conf)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])
  normalizing_constant = -(conf * tf.math.log(conf) +
      (effective_vocab_size - 1) * low_conf * tf.math.log(low_conf + 1e-20))

  cross_entropy -= normalizing_constant
  weights = tf.cast(tf.not_equal(labels, 0), 'float32')
  cross_entropy *= weights
  return tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule."""
  def __init__(self, learning_rate, hidden_size, warmup_steps):
    """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._hidden_size = hidden_size
    self._warmup_steps = tf.cast(warmup_steps, 'float32')

  def __call__(self, global_step):
    """Computes learning rate with linear warmup and rsqrt decay.

    Args:
      global_step: int scalar tensor, the current global step. 

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`. 
    """
    global_step = tf.cast(global_step, 'float32')
    learning_rate = self._learning_rate
    learning_rate *= (self._hidden_size**-0.5)
    # linear warmup
    learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
    # rsqrt decay
    learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
    return learning_rate

def create_optimizer(learning_rate, beta1, beta2, epsilon):
  """Creates Adam optimizer.

  Args:
    learning_rate: float scalar, base learning rate.
    beta1: float scalar, beta1 parameter of Adam optimizer.
    beta2: float scalar, beta2 parameter of Adam optimizer.
    epsilon: float scalar, epsilon parameter of Adam optimizer.

  Returns:
    optimizer: an instance of tf.keras.optimizer.Adam. 
  """
  optimizer = tf.keras.optimizers.Adam(learning_rate, 
                                       beta1, 
                                       beta2, 
                                       epsilon=epsilon)
  return optimizer


def save_attention_weights(filename, data):
  """Saves attention weights data to *.npy file.

  Args:
    filename: string scalar, filename.
    data: a list or tuple or dict of numpy arrays, the attention weights and 
      token ids of input and translated sequence.
  """
  np.save(filename, data)
