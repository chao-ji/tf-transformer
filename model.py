"""Defines Transformation model in keras API.
"""
import tensorflow as tf

import beam_search 
import utils

NEG_INF = -1e9
SOS_ID = 0
EOS_ID = 1


class Projection(tf.keras.layers.Layer):
  """Linearly projects a batch of continuously represented query or reference 
  sequences.

  The projection operates in either "split" mode or "merge" mode:

    - "split" mode converts the input sequences in the original representation 
      into the multi-headed "query", "key" or "value" for the attention 
      computation. 

      Input: [batch_size(N), seq_len(T), hidden_size(D)] 
      Weight: [hidden_size(D), num_heads(H), size_per_head(S)]
      Output: dot([N*T, D], [D, H*S]) reshape ==> [N, T, H, S]

    - "merge" mode is the opposite of "split", which converts the multi-headed 
      "value" back to the original representation.

      Input: [batch_size(N), seq_len(T), num_heads(H), size_per_head(S)]
      Weight: [num_heads(H), size_per_head(S), hidden_size(D)]
      Output: dot([N*T, H*S], [H*S, D]) reshape ==> [N, T, D]
  """
  def __init__(self, 
               num_heads, 
               size_per_head, 
               kernel_initializer='glorot_uniform', 
               mode="split"):
    """Constructor.

    Args:
      num_heads: int scalar, num of attention heads.
      size_per_head: int scalar, the hidden size of each attention head.
      kernel_initializer: string scalar, the weight initializer.
      mode: string scalar, mode of projection ("split" or "merge") . 
    """
    super(Projection, self).__init__()
    if mode not in ('split', 'merge'):
      raise ValueError('"mode" must be either "split" or "merge".')
    self._num_heads = num_heads
    self._size_per_head = size_per_head
    self._hidden_size = num_heads * size_per_head
    self._kernel_initializer = kernel_initializer
    self._mode = mode

  def build(self, inputs_shape):
    """Creates variables of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor, the last element 
        corresponds to the depth.
    """
    depth = inputs_shape[-1]
    if depth is None: 
      raise ValueError('The depth of inputs must be not be None.')

    if self._mode == 'merge':
      kernel_shape = (self._num_heads, self._size_per_head, self._hidden_size)
    else:
      kernel_shape = (self._hidden_size, self._num_heads, self._size_per_head)

    self.add_weight(name='kernel',
                    shape=kernel_shape,
                    initializer=self._kernel_initializer,
                    dtype='float32',
                    trainable=True)
    super(Projection, self).build(inputs_shape)

  def call(self, inputs):
    """Performs the projection.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, num_heads, 
        size_per_head] in "merge" mode, or float tensor of shape [batch_size, 
        seq_len, hidden_size] in "split" mode.

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size] in 
        "merge" mode, or float tensor of shape [batch_size, seq_len, num_heads, 
        size_per_head] int "split" mode.
    """
    kernel = self.trainable_variables[0] 
    if self._mode == 'merge':
      outputs = tf.einsum('NTHS,HSD->NTD', inputs, kernel)
    else:
      outputs = tf.einsum('NTD,DHS->NTHS', inputs, kernel)
    return outputs


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer.

  Given a batch of query sequences (tensor of shape [batch_size, q_seq_len, 
  hidden_size]) and a batch of reference sequences (tensor of shape [batch_size,
  r_seq_len, hidden_size]) in continuous representation, this layer computes a 
  new representation of the query sequences by making them selectively attend to
  different tokens in reference sequences.
 
  When the query and reference sequences are the same, this ends up being "Self 
  Attention", i.e. query sequences attend to themselves. 
  """
  def __init__(self, hidden_size, num_heads, dropout_rate):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
    """
    super(Attention, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._size_per_head = hidden_size // num_heads
    
    self._dense_layer_query = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_key = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_value = Projection(
        num_heads, self._size_per_head, mode='split')
    self._dense_layer_output = Projection(
        num_heads, self._size_per_head, mode='merge')
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, query, reference, padding_mask, training, cache=None):
    """Computes new representation of `query` by attending to `reference`.

    Args:
      query: float tensor of shape [batch_size, q_seq_len, hidden_size], query 
        sequences.
      reference: float tensor of shape [batch_size, r_seq_len, hidden_size], 
        reference sequences.
      padding_mask: float tensor of shape [batch_size, 1, 1, r_seq_len], 
        populated with either 0 (for regular tokens) or `NEG_INF` (in order to 
        mask out padded tokens).
      training: bool scalar, True if in training mode.
      cache: None, or dict with entries
        'k': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head],
        'v': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head].

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the 
        new representation of `query`.
    """
    # [batch_size, q_seq_len, num_heads, size_per_head]
    query = self._dense_layer_query(query)
    query *= self._size_per_head ** -0.5 

    # [batch_size, r_seq_len, num_heads, size_per_head]
    key = self._dense_layer_key(reference)

    # [batch_size, r_seq_len, num_heads, size_per_head]
    value = self._dense_layer_value(reference)
    
    if cache is not None:
      # concatenate along the `seq_len` dimension
      cache['k'] = key = tf.concat([cache['k'], key], axis=1)
      cache['v'] = value = tf.concat([cache['v'], value], axis=1)

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    similarities = tf.einsum('NQHS,NRHS->NHQR', query, key)

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    similarities += padding_mask * NEG_INF

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    similarities = tf.nn.softmax(similarities, axis=3)
    similarities = self._dropout_layer(similarities, training=training)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    outputs = tf.einsum('NHQR,NRHS->NQHS', similarities, value) 

    # [batch_size, q_seq_len, hidden_size]
    outputs = self._dense_layer_output(outputs) 
    return outputs


class FeedForwardNetwork(tf.keras.layers.Layer):
  """The Projection layer that consists of a tandem of two dense layers (an
  intermediate layer and an output layer).
  """
  def __init__(self, hidden_size, filter_size, dropout_rate):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation, 
        which is also the depth of the output dense layer.
      filter_size: int scalar, the depth of the intermediate dense layer. 
      dropout_rate: float scalar, dropout rate for the Dropout layers. 
    """
    super(FeedForwardNetwork, self).__init__()
    self._hidden_size = hidden_size
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._dense_layer_filter = tf.keras.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu)
    self._dense_layer_output = tf.keras.layers.Dense(hidden_size, use_bias=True)
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training):
    """Performs projection through two dense layers.

    Args:
      inputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        input sequences.
      training: bool scalar, True if in training mode.

    Return:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size], the 
        output sequences.
    """
    outputs = self._dense_layer_filter(inputs)
    outputs = self._dropout_layer(outputs, training=training)
    outputs = self._dense_layer_output(outputs)
    return outputs


class EmbeddingLayer(tf.keras.layers.Layer):
  """The customized layer that operates in "embedding" mode, which converts 
  token ids to embedding vectors, or "logits" mode, which converts embedding 
  vectors to logits, reusing the same weight matrix used in "embedding" mode. 

  Note that the Encoder only performs one-way conversion (source language token
  ids to embedding vectors), while the Decoder performs two-way conversion (
  target language token ids to embedding vector, and embedding vector to logits
  that precedes the softmax, which reuses the weights). 
  """
  def __init__(self, vocab_size, hidden_size):
    """Constructor.

    Args:
      vocab_size: int scalar, num of tokens (including SOS and EOS) in the 
        vocabulary.
      hidden_size: int scalar, the hidden size of continuous representation.
    """
    super(EmbeddingLayer, self).__init__()
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self.add_weight('weights', 
                    shape=[vocab_size, hidden_size], 
                    initializer=tf.keras.initializers.RandomNormal(
                        mean=0., stddev=hidden_size ** -0.5))

  def call(self, inputs, mode):
    """Either converts token ids to embeddings, or embeddings to logits.

    Args:
      inputs: int tensor of shape [batch_size, seq_len] in "embedding" mode, the
        sequences token ids; or float tensor of shape [batch_size, seq_len, 
        hidden_size] in "logits" mode, the sequences in continuous 
        representation.
      mode: string scalar, "embedding" or "logits".

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size] in 
        "embedding" mode, the sequences in continuous representation; or float 
        tensor of shape [batch_size, seq_len, vocab_size] in "logits" mode, the
        logits preceding the softmax.
    """
    if mode == 'embedding':
      outputs = self._tokens_to_embeddings(inputs)
    elif mode == 'logits':
      outputs = self._embeddings_to_logits(inputs)
    else:
      raise ValueError('Invalid mode {}'.format(mode))
    return outputs

  def _get_vocab_embeddings(self):
    """Returns the embedding matrix ([vocab_size, hidden_size])

    Note the SOS token has a fixed (not trainable) all-zero embedding vector.
    """
    return tf.pad(self.weights[0][1:], [[1, 0], [0, 0]]) 

  def _tokens_to_embeddings(self, inputs):
    """The dense layer that converts token IDs to embedding vectors.

    Args:
      inputs: int tensor of shape [batch_size, seq_len], the sequences token 
        ids.

    Returns:
      embeddings: float tensor of shape [batch_size, seq_len, hidden_size], the
        sequences in continuous representation.
    """
    # [vocab_size, hidden_size]
    embeddings = self._get_vocab_embeddings()

    # [batch_size, src_seq_len, hidden_size]
    embeddings = tf.gather(embeddings, inputs)

    embeddings *= self._hidden_size ** 0.5
    embeddings = tf.cast(embeddings, 'float32')
    return embeddings
    
  def _embeddings_to_logits(self, decoder_outputs):
    """The dense layer preceding the softmax that computes the logits.

    Args:
      decoder_outputs: float tensor of shape [batch_size, tgt_seq_len, 
        hidden_size], the sequences in continuous representation.

    Returns:
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
        logits preceding the softmax.
    """
    # [vocab_size, hidden_size]
    embeddings = self._get_vocab_embeddings()
    batch_size = tf.shape(decoder_outputs)[0]
    tgt_seq_len = tf.shape(decoder_outputs)[1]

    # [batch_size * tgt_seq_len, hidden_size]
    decoder_outputs = tf.reshape(decoder_outputs, [-1, self._hidden_size])
    logits = tf.matmul(decoder_outputs, embeddings, transpose_b=True)
    logits = tf.reshape(logits, [batch_size, tgt_seq_len, self._vocab_size])
    return logits


class EncoderLayer(tf.keras.layers.Layer):
  """The building block that makes the encoder stack of layers, consisting of an
  attention sublayer and a feed-forward sublayer.
  """
  def __init__(self, hidden_size, num_heads, filter_size, dropout_rate):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer. 
      dropout_rate: float scalar, dropout rate for the Dropout layers.
    """
    super(EncoderLayer, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._mha = Attention(hidden_size, num_heads, dropout_rate)
    self._layernorm_mha = tf.keras.layers.LayerNormalization() 
    self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

    self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
    self._layernorm_ffn = tf.keras.layers.LayerNormalization()
    self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, padding_mask, training):
    """Computes the output of the encoder layer.

    Args:
      inputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
        input source sequences.
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for regular tokens) or `NEG_INF` (in order to
        mask out padded tokens).
      training: bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
        output source sequences.
    """
    query = refernce = self._layernorm_mha(inputs)
    outputs = self._mha(query, refernce, padding_mask, training)
    ffn_inputs = self._dropout_mha(outputs, training=training) + inputs

    outputs = self._layernorm_ffn(ffn_inputs)
    outputs = self._ffn(outputs, training)
    outputs = self._dropout_ffn(outputs, training=training) + ffn_inputs
    return outputs


class DecoderLayer(tf.keras.layers.Layer):
  """The building block that makes the decoder stack of layers, consisting of a 
  self-attention sublayer, cross-attention sublayer and a feed-forward sublayer.
  """
  def __init__(self, hidden_size, num_heads, filter_size, dropout_rate):
    """Constructor.

    Args:
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
    """
    super(DecoderLayer, self).__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._mha_intra = Attention(hidden_size, num_heads, dropout_rate)
    self._layernorm_mha_intra = tf.keras.layers.LayerNormalization() 
    self._dropout_mha_intra = tf.keras.layers.Dropout(dropout_rate)

    self._mha_inter = Attention(hidden_size, num_heads, dropout_rate)
    self._layernorm_mha_inter = tf.keras.layers.LayerNormalization() 
    self._dropout_mha_inter = tf.keras.layers.Dropout(dropout_rate) 

    self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
    self._layernorm_ffn = tf.keras.layers.LayerNormalization()
    self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(self, 
           inputs, 
           encoder_outputs, 
           look_ahead_mask, 
           padding_mask, 
           training, 
           cache=None):
    """Computes the output of the decoder layer.

    Args:
      inputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
        input target sequences.
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size], the encoded source sequences to be used as reference.
      look_ahead_mask: float tensor of shape [1, 1, tgt_seq_len, tgt_seq_len], 
        populated with either 0 (for current or past tokens) or `NEG_INF` (in
        order to mask out future tokens in target sequences).
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for regular tokens) or `NEG_INF` (in order to
        mask out padded tokens in source sequences).
      training: bool scalar, True if in training mode.
      cache: None, or dict with entries
        'k': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head],
        'v': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head]

    Returns:
      outputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
        output target sequences.
    """
    query = reference = self._layernorm_mha_intra(inputs)
    outputs = self._mha_intra(
        query, reference, look_ahead_mask, training, cache=cache)
    mha_inter_inputs = self._dropout_mha_intra(outputs, training=training
        ) + inputs

    query, reference = self._layernorm_mha_inter(mha_inter_inputs
        ), encoder_outputs
    outputs = self._mha_inter(query, reference, padding_mask, training)
    ffn_inputs = self._dropout_mha_inter(outputs, training=training
        ) + mha_inter_inputs

    outputs = self._layernorm_ffn(ffn_inputs)
    outputs = self._ffn(outputs, training)
    outputs = self._dropout_ffn(outputs, training=training) + ffn_inputs
    return outputs


class Encoder(tf.keras.layers.Layer):
  """The Encoder that consists of a stack of structurally identical layers."""
  def __init__(
      self, stack_size, hidden_size, num_heads, filter_size, dropout_rate):
    """Constructor.

    Args:
      stack_size: int scalar, num of layers in the stack.
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer. 
      dropout_rate: float scalar, dropout rate for the Dropout layers.     
    """
    super(Encoder, self).__init__()
    self._stack_size = stack_size 
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._stack = [EncoderLayer(hidden_size, 
                                num_heads, 
                                filter_size, 
                                dropout_rate) for _ in range(self._stack_size)]
    self._layernorm = tf.keras.layers.LayerNormalization() 

  def call(self, inputs, padding_mask, training):
    """Computes the output of the encoder stack of layers. 

    Args:
      inputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
        input source sequences.
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for regular tokens) or `NEG_INF` (in order to
        mask out padded tokens).
      training: bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
        output source sequences.
    """
    for layer in self._stack:
      inputs = layer.call(inputs, padding_mask, training)
    outputs = self._layernorm(inputs)
    return outputs


class Decoder(tf.keras.layers.Layer):
  """Decoder that consists of a stack of structurally identical layers."""
  def __init__(
      self, stack_size, hidden_size, num_heads, filter_size, dropout_rate):
    """Constructor.

    Args:
      stack_size: int scalar, the num of layers in the stack.
      hidden_size: int scalar, the hidden size of continuous representation.
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer. 
      dropout_rate: float scalar, dropout rate for the Dropout layers.      
    """
    super(Decoder, self).__init__()
    self._stack_size = stack_size 
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate

    self._stack = [DecoderLayer(
        hidden_size, num_heads, filter_size, dropout_rate) 
        for _ in range(self._stack_size)]
    self._layernorm = tf.keras.layers.LayerNormalization() 
 
  def call(self, 
           inputs, 
           encoder_outputs, 
           look_ahead_mask, 
           padding_mask, 
           training, 
           cache=None):
    """Computes the output of the decoder stack of layers.

    Args:
      inputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
        input target sequences.
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size], the encoded source sequences to be used as refernece.
      look_ahead_mask: float tensor of shape [1, 1, tgt_seq_len, tgt_seq_len], 
        populated with either 0 (for current or past tokens) or `NEG_INF` (in
        order to mask out future tokens in target sequences).
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for regular tokens) or `NEG_INF` (in order to
        mask out padded tokens in source sequences).
      training: bool scalar, True if in training mode.
      cache: None, or a dict with keys 'layer_0', ... 
        'layer_[self.num_layers - 1]', where the value
        associated with each key is a dict with entries
          'k': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
            size_per_head],
          'v': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
            size_per_head]. 

    Returns:
      outputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
        output target sequences.
    """
    for i, layer in enumerate(self._stack):
      inputs = layer.call(inputs, 
                          encoder_outputs, 
                          look_ahead_mask, 
                          padding_mask, 
                          training, 
                          cache=cache['layer_%d' % i] 
                              if cache is not None else None)
    outputs = self._layernorm(inputs)
    return outputs


class TransformerModel(tf.keras.Model):
  """Transformer model as described in https://arxiv.org/abs/1706.03762

  The model implements methods `call` and `transduce`, where
    - `call` takes as input BOTH the source and target token ids, and returns 
      the estimated logits for the target token ids.
    - `transduce` takes as input the source token ids ONLY, and outputs the 
      token ids of the decoded target sequences using beam search. 
  """
  def __init__(self, 
               encoder_stack_size=6, 
               decoder_stack_size=6, 
               hidden_size=512, 
               num_heads=8, 
               filter_size=2048, 
               vocab_size=33945,
               dropout_rate=0.1,
               extra_decode_length=50,
               beam_width=4,
               alpha=0.6):
    """Constructor.

    Args:
      encoder_stack_size: int scalar, num of layers in encoder stack.
      decoder_stack_size: int scalar, num of layers in decoder stack.
      hidden_size: int scalar, the hidden size of continuous representation. 
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      vocab_size: int scalar, num of subword tokens (including SOS and EOS/PAD) 
        in the vocabulary. 
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      extra_decode_length: int scalar, the max decode length would be the sum of
        `tgt_seq_len` and `extra_decode_length`.
      beam_width: int scalar, beam width for beam search.
      alpha: float scalar, the parameter for length normalization used in beam 
        search.
    """
    super(TransformerModel, self).__init__()
    self._encoder_stack_size = encoder_stack_size
    self._decoder_stack_size = decoder_stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._vocab_size = vocab_size
    self._dropout_rate = dropout_rate
    self._extra_decode_length = extra_decode_length
    self._beam_width = beam_width
    self._alpha = alpha

    self._embedding_logits_layer = EmbeddingLayer(vocab_size, hidden_size)
    self._encoder = Encoder(
        encoder_stack_size, hidden_size, num_heads, filter_size, dropout_rate)
    self._decoder = Decoder(
        decoder_stack_size, hidden_size, num_heads, filter_size, dropout_rate)

    self._encoder_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._decoder_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, src_token_ids, tgt_token_ids, training=False):
    """Takes as input the source and target token ids, and returns the estimated
    logits for the target sequences.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.
      tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len], token ids 
        of target sequences
      training: bool scalar, True if in training mode (i.e. dropout is on).

    Returns:
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size]. 
    """
    padding_mask = utils.get_padding_mask(src_token_ids)
    encoder_outputs = self._encode(src_token_ids, padding_mask, training)
    logits = self._decode(tgt_token_ids, encoder_outputs, padding_mask, training)
    return logits

  def _encode(self, src_token_ids, padding_mask, training=False):
    """Convert source sequences token ids into continuous representation, and 
    compute the Encoder-encoded sequences.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for regular tokens) or `NEG_INF` (for padded
        tokens).
      training: bool scalar, True if in training mode.

    Returns:
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size], the encoded source sequences to be used as refernece. 
    """
    # [batch_size, src_seq_len, hidden_size]
    src_token_embeddings = self._embedding_logits_layer(
        src_token_ids, 'embedding')
    src_seq_len = tf.shape(src_token_ids)[1]

    # [src_seq_len, hidden_size]
    positional_encoding = utils.get_positional_encoding(
        src_seq_len, self._hidden_size)

    src_token_embeddings += positional_encoding
    src_token_embeddings = self._encoder_dropout_layer(
        src_token_embeddings, training)

    encoder_outputs = self._encoder(
        src_token_embeddings, padding_mask, training)
    return encoder_outputs

  def _decode(
      self, tgt_token_ids, encoder_outputs, padding_mask, training=False):
    """Compute the estimated logits of target token ids, based on the encoded 
    source sequences.

    Args:
      tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len] in training 
        mode, token ids of target sequences; or None in inference mode.
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size], the encoded source sequences to be used as refernece. 
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for regular tokens) or `NEG_INF` (in order to
        mask out padded tokens in source sequences).      

    Returns:
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size].
    """
    tgt_seq_len = tf.shape(tgt_token_ids)[1]

    # [batch_size, tgt_seq_len, hidden_size]
    tgt_token_embeddings = self._embedding_logits_layer(
        tgt_token_ids, 'embedding')

    positional_encoding = utils.get_positional_encoding(
        tgt_seq_len, self._hidden_size)
    tgt_token_embeddings += positional_encoding
    tgt_token_embeddings = self._decoder_dropout_layer(
        tgt_token_embeddings, training) 

    look_ahead_mask = utils.get_look_ahead_mask(tgt_seq_len)

    decoder_outputs = self._decoder(tgt_token_embeddings, 
                                    encoder_outputs, 
                                    look_ahead_mask, 
                                    padding_mask, 
                                    training)

    logits = self._embedding_logits_layer(decoder_outputs, 'logits')
    return logits

  def transduce(self, src_token_ids):
    """Takes as input the source token ids only, and outputs the token ids of 
    the decoded target sequences using beam search.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.

    Returns:
      decoded_ids: int tensor of shape [batch_size, decoded_seq_len], the token
        ids of the decoded target sequences using beam search.
      scores: float tensor of shape [batch_size], the scores (length-normalized 
        log-probs) of the decoded target sequences.
    """
    batch_size, src_seq_len = tf.unstack(tf.shape(src_token_ids))
    max_decode_length = src_seq_len + self._extra_decode_length

    decoding_fn = self._build_decoding_fn(max_decode_length)
    decoding_cache = self._build_decoding_cache(src_token_ids, batch_size)
    sos_ids = tf.ones([batch_size], dtype='int32') * SOS_ID

    bs = beam_search.BeamSearch(decoding_fn, 
                                self._embedding_logits_layer._vocab_size, 
                                batch_size,
                                self._beam_width, 
                                self._alpha, 
                                max_decode_length, 
                                EOS_ID)
    decoded_ids, scores = bs.search(sos_ids, decoding_cache)

    decoded_ids = decoded_ids[:, 0, 1:]
    scores = scores[:, 0] 
    return decoded_ids, scores 

  def _build_decoding_cache(self, src_token_ids, batch_size):
    """Builds a dictionary that caches previously computed key and value feature
    maps of the growing decoded sequence.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids of 
        source sequences. 
      batch_size: int scalar, num of sequences in a batch.

    Returns:
      decoding_cache: dict of entries
          'encoder_outputs': tensor of shape [batch_size, src_seq_len, 
            hidden_size],
          'padding_mask': tensor of shape [batch_size, 1, 1, src_seq_len],

          and entries with keys 'layer_0',...,'layer_[decoder_num_layers - 1]'
          where the value associated with key 'layer_*' is a dict with entries
            'k': tensor of shape [batch_size, 0, num_heads, size_per_head],
            'v': tensor of shape [batch_size, 0, num_heads, size_per_head].
    """
    padding_mask = utils.get_padding_mask(src_token_ids)
    encoder_outputs = self._encode(src_token_ids, padding_mask, training=False)
    size_per_head = self._hidden_size // self._num_heads

    decoding_cache = {'layer_%d' % layer:
        {'k':
            tf.zeros([
                batch_size, 0, self._num_heads, size_per_head
            ], 'float32'),
         'v':
            tf.zeros([
                batch_size, 0, self._num_heads, size_per_head
            ], 'float32')
        } for layer in range(self._decoder._stack_size)
    }
    decoding_cache["encoder_outputs"] = encoder_outputs
    decoding_cache["padding_mask"] = padding_mask
    return decoding_cache

  def _build_decoding_fn(self, max_decode_length):
    """Builds the decoding function that computs the decoded sequences using
    beam search.

    Args:
      max_decode_length: int scalar, the decoded sequences would not exceed
        `max_decode_length`.

    Returns:
      decoding_fn: a callable that outputs the logits of the next decoded token
        ids.
    """
    timing_signal = utils.get_positional_encoding(
        max_decode_length, self._hidden_size)
    timing_signal = tf.cast(timing_signal, 'float32') 

    def decoding_fn(decoder_input, i, cache):
      """Computes the logits of the next decoded token ids.

      Args:
        decoder_input: int tensor of shape [batch_size * beam_size, 1], the 
          decoded tokens at index `i`.
        i: int scalar tensor, the index of the `decoder_input` in the decoded
          sequence. 
        cache: dict of entries
          'encoder_outputs': tensor of shape 
            [batch_size * beam_width, src_seq_len, hidden_size],
          'padding_mask': tensor of shape
            [batch_size * beam_width, 1, 1, src_seq_len],

          and entries with keys 'layer_0',...,'layer_[decoder_num_layers - 1]'
          where the value associated with key 'layer_*' is a dict with entries
            'k': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
              size_per_head],
            'v': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
              size_per_head].

      Returns:
        logits: float tensor of shape [batch_size * beam_size, vocab_size].
        cache: a dict with the same structure as the input `cache`, except that
          the shapes of the values of key `k` and `v` are
          [batch_size * beam_width, seq_len + 1, num_heads, size_per_head].
      """
      decoder_input = self._embedding_logits_layer(decoder_input, 'embedding')
      decoder_input += timing_signal[i:i + 1]

      decoder_outputs = self._decoder(decoder_input,
                                      cache['encoder_outputs'],
                                      tf.zeros((1, i + 1), dtype='float32'),
                                      cache['padding_mask'],
                                      training=False,
                                      cache=cache)

      logits = self._embedding_logits_layer(decoder_outputs, mode='logits')
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return decoding_fn 
