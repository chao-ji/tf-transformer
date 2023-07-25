"""Defines utility functions."""
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


def get_positional_encoding(seq_len, hidden_size):
  """Creates a tensor that encodes positional information.

  Args:
    seq_len: int scalar tensor, sequence length.
    hidden_size: int scalar, the hidden size of continuous representation. 

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


def compute_loss(labels, logits, smoothing, vocab_size, padding_value=0):
  """Computes average (per-token) cross entropy loss.

  1. Applies label smoothing -- all entries in the groundtruth label tensor
     get non-zero probability mass.
  2. Computes per token loss of shape [batch_size, tgt_seq_len], where padded
     positions are masked, and then the sum of per token loss is normalized by
     the total number of non-padding entries.

  Args:
    labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
      token ids.
    logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
      predicted logits of tokens over the vocabulary.
    smoothing: float scalar, the amount of label smoothing applied to the
      one-hot class labels.
    vocab_size: int scalar, num of tokens (including SOS and EOS) in the
      vocabulary.
    padding_value: int scalar, the vocabulary index of the PAD token.

  Returns:
    loss: float scalar tensor, the per-token cross entropy
  """
  # effective_vocab = vocab - {SOS_ID}
  effective_vocab_size = vocab_size - 1

  # prob mass allocated to the token that should've been predicted
  on_value = 1.0 - smoothing
  # prob mass allocated to all other tokens
  off_value = smoothing / (effective_vocab_size - 1)

  # [batch_size, tgt_seq_len, vocab_size]
  labels_one_hot = tf.one_hot(
      labels,
      depth=vocab_size,
      on_value=on_value,
      off_value=off_value)

  # compute cross entropy over all tokens in vocabulary but SOS_ID (i.e. 0)
  # because SOS_ID should never appear in the decoded sequence
  # [batch_size, tgt_seq_len]
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])

  # this is the entropy when the softmax'ed logits == groundtruth labels
  # so it should be deducted from `cross_entropy` to make sure the minimum
  # possible cross entropy == 0
  normalizing_constant = -(on_value * tf.math.log(on_value) +
      (effective_vocab_size - 1) * off_value * tf.math.log(off_value + 1e-20))
  cross_entropy -= normalizing_constant

  # mask out predictions where the labels == `padding_value`
  weights = tf.cast(tf.not_equal(labels, padding_value), 'float32')
  cross_entropy *= weights
  loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)
  return loss
