"""Defines utility functions."""
import numpy as np
import tensorflow as tf


def get_padding_mask(inputs, padding_value=0):
  """Creates a binary tensor to mask out padded tokens.

  Args:
    inputs: int tensor of shape [batch_size, src_seq_len], token ids
      of source sequences.
    padding_value: int scalar, the vocabulary index of the PAD token. 

  Returns:
    mask: binary tensor of shape [batch_size, 1, 1, src_seq_len], storing ones
      for padded tokens and zeros for regular tokens.
  """
  mask = tf.cast(tf.equal(inputs, padding_value), 'float32') 
  mask = mask[:, tf.newaxis, tf.newaxis, :]
  return mask
 
 
def get_look_ahead_mask(seq_len):
  """Creates a tensor to mask out future tokens in the target sequences when in 
  training mode.

  Given sequence length `L` of target sequence, the mask would be a L x L
  matrix (when `tf.squeeze`'ed) where upper diagonal entries are ones and all 
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


def get_positional_encoding(seq_len, hidden_size, reverse=False):
  """Creates a tensor that encodes positional information.

  Args:
    seq_len: int scalar tensor, sequence length.
    hidden_size: int scalar, the hidden size of continuous representation. 
    reverse: bool, whether to reverse the sequence. Defaults to False.

  Returns:
    positional_encoding: float tensor of shape [seq_len, hidden_size], the 
      tensor that encodes positional information.
  """
  distances = tf.cast(tf.range(seq_len), 'float32')
  hidden_size //= 2
  inverse_frequencies = 1 / (
      10000 ** (tf.cast(tf.range(hidden_size), 'float32') / (hidden_size - 1)))
  positional_encoding = tf.einsum('i,j->ij', distances, inverse_frequencies)
  positional_encoding = tf.concat([tf.sin(positional_encoding),
                                   tf.cos(positional_encoding)], axis=1)
  return positional_encoding
