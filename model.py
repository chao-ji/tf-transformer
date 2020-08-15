"""Defines Transformer model in tf.keras API."""
import tensorflow as tf

import utils

from commons.tokenization import SOS_ID
from commons.tokenization import EOS_ID
from commons.beam_search import NEG_INF
from commons import beam_search


class Projection(tf.keras.layers.Layer):
  """Linearly projects a batch of continuously represented sequences of tokens.

  This projection layer operates in either Split mode or Merge mode:

    - Split mode converts the input sequences in the original representation 
      into the multi-headed "query", "key" or "value" for the attention 
      computation. 

      Input: [batch_size(N), seq_len(T), hidden_size(D)] 
      Weight: [hidden_size(D), num_heads(H), size_per_head(S)]
      Output: dot([N*T, D], [D, H*S]) reshape ==> [N, T, H, S]

    - Merge mode performs the opposite action of Split, converting the 
      multi-headed "value" back to the original representation.

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
    """Creates weights of this layer.

    Args:
      inputs_shape: tuple of ints or 1-D int tensor, the last element 
        corresponds to the depth.
    """
    depth = inputs_shape[-1]
    if depth is None: 
      raise ValueError('The depth of inputs must not be None.')

    if self._mode == 'merge':
      kernel_shape = self._num_heads, self._size_per_head, self._hidden_size
    else:
      kernel_shape = self._hidden_size, self._num_heads, self._size_per_head

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
        size_per_head] in Merge mode, or float tensor of shape [batch_size, 
        seq_len, hidden_size] in Split mode.

    Returns:
      outputs: float tensor of shape [batch_size, seq_len, hidden_size] in 
        Merge mode, or float tensor of shape [batch_size, seq_len, num_heads, 
        size_per_head] int Split mode.
    """
    kernel = self.trainable_variables[0]
    if self._mode == 'merge':
      outputs = tf.einsum('NTHS,HSD->NTD', inputs, kernel)
    else:
      outputs = tf.einsum('NTD,DHS->NTHS', inputs, kernel)
    return outputs


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer.

  Given a batch of continuously represented query sequences (tensor of shape [
  batch_size, q_seq_len, hidden_size]) and reference sequences (tensor of shape 
  [batch_size, r_seq_len, hidden_size]), this layer computes a new 
  representation of the query sequences by making them selectively attend to
  different tokens in reference sequences.
 
  When the query and reference sequence are the same, this layer ends up being 
  "Self Attention" -- the query sequence attends to itself. 
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

    self._attention_weights = None

  def call(self, query_seqs, reference_seqs, token_mask, training, cache=None):
    """Computes new representation of query sequences.

    Args:
      query_seqs: float tensor of shape [batch_size, q_seq_len, hidden_size], 
        query sequences.
      reference_seqs: float tensor of shape [batch_size, r_seq_len, hidden_size]
        , reference sequences.
      token_mask: float tensor of shape [batch_size, num_heads, q_seq_len, 
        r_seq_len], populated with either 0 (for tokens to keep) or 1 (for 
        tokens to be masked).
      training: bool scalar, True if in training mode.
      cache: (Optional) dict with entries
        'k': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head],
        'v': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head],
        'tgt_tgt_attention': tensor of shape [batch_size * beam_width, 
          num_heads, tgt_seq_len, tgt_seq_len],
        'tgt_src_attention': tensor of shape [batch_size * beam_width, 
          num_heads, tgt_seq_len, src_seq_len].
        Must be provided in inference mode when called within decoder layers.

    Returns:
      outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the 
        new representation of `query_seqs`.
    """
    self_attention = True if id(query_seqs) == id(reference_seqs) else False
    
    # [batch_size, q_seq_len, num_heads, size_per_head]
    query = self._dense_layer_query(query_seqs)
    query *= self._size_per_head ** -0.5 

    # [batch_size, r_seq_len, num_heads, size_per_head]
    key = self._dense_layer_key(reference_seqs)

    # [batch_size, r_seq_len, num_heads, size_per_head]
    value = self._dense_layer_value(reference_seqs)
    
    if cache is not None and self_attention:
      # concatenate along the `seq_len` dimension
      cache['k'] = key = tf.concat([cache['k'], key], axis=1)
      cache['v'] = value = tf.concat([cache['v'], value], axis=1)

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    attention_weights = tf.einsum('NQHS,NRHS->NHQR', query, key)

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    attention_weights += token_mask * NEG_INF

    # [batch_size, num_heads, q_seq_len, r_seq_len]
    attention_weights = tf.nn.softmax(attention_weights, axis=3)
    attention_weights = self._dropout_layer(
        attention_weights, training=training)

    # save attention weights of encoder layers in inference mode
    if not training and cache is None and self_attention:
      self._attention_weights = attention_weights 

    # save attention weights for visualization in inference mode
    if cache is not None:
      if self_attention:
        # [batch_size, num_heads, tgt_seq_len, tgt_seq_len]
        cache['tgt_tgt_attention'] = tf.concat([tf.pad(
            cache['tgt_tgt_attention'], [[0, 0], [0, 0], [0, 0], [0, 1]]),
            attention_weights], axis=2)
      else:
        # [batch_size, num_heads, tgt_src_len, src_seq_len]
        cache['tgt_src_attention'] = tf.concat([
            cache['tgt_src_attention'], attention_weights], axis=2)

    # [batch_size, q_seq_len, num_heads, size_per_head]
    outputs = tf.einsum('NHQR,NRHS->NQHS', attention_weights, value) 

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
  """The customized layer that operates in Embedding mode or Logits mode.

  - Embedding mode converts token ids to embedding vectors.
    Input: [batch_size(N), seq_len(T)]
    Weight: [vocab_size(V), hidden_size(D)]
    Output: [batch_size(N), seq_len(T), hidden_size(D)]
  
  - Logits mode converts embedding vectors to logits.
    Input: [batch_size(N), seq_len(T), hidden_size(D)]
    Weight: [vocab_size(V), hidden_size(D)]
    Output: [batch_size(N), seq_len(T), vocab_size(V)]

  Note that Logits mode reuses the same weight matrix in Embedding mode.
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
    """Returns the embedding matrix (of shape [vocab_size, hidden_size]). Note 
    that SOS token (index 0) has a fixed (not trainable) zero embedding vector.
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

    # [batch_size, seq_len, hidden_size]
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
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked).
      training: bool scalar, True if in training mode.

    Returns:
      outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
        output source sequences.
    """
    query = reference = self._layernorm_mha(inputs)
    outputs = self._mha(query, reference, padding_mask, training)
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
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked). 
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked). 
      training: bool scalar, True if in training mode.
      cache: (Optional) dict with entries
        'k': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head],
        'v': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
          size_per_head],
        'tgt_tgt_attention': tensor of shape [batch_size * beam_width, 
          num_heads, tgt_seq_len, tgt_seq_len],
        'tgt_src_attention': tensor of shape [batch_size * beam_width, 
          num_heads, tgt_seq_len, src_seq_len].
        Must be provided in inference mode.

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
    outputs = self._mha_inter(
        query, reference, padding_mask, training, cache=cache)
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
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked). 
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
        hidden_size], the encoded source sequences to be used as reference.
      look_ahead_mask: float tensor of shape [1, 1, tgt_seq_len, tgt_seq_len], 
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked). 
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked). 
      training: bool scalar, True if in training mode.
      cache: (Optional) dict with keys 'layer_0', ... 
        'layer_[self.num_layers - 1]', where the value
        associated with each key is a dict with entries
          'k': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
            size_per_head],
          'v': tensor of shape [batch_size * beam_width, seq_len, num_heads, 
            size_per_head],
          'tgt_tgt_attention': tensor of shape [batch_size * beam_width, 
            num_heads, tgt_seq_len, tgt_seq_len],
          'tgt_src_attention': tensor of shape [batch_size * beam_width, 
            num_heads, tgt_seq_len, src_seq_len]. 
        Must be provided in inference mode.

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
    - `call` is invoked in training mode, taking as input BOTH the source and 
      target token ids, and returning the estimated logits for the target token 
      ids.
    - `transduce` is invoked in inference mode, taking as input the source token 
      ids ONLY, and outputting the token ids of the decoded target sequences 
      using beam search. 
  """
  def __init__(self, 
               vocab_size,
               encoder_stack_size=6, 
               decoder_stack_size=6, 
               hidden_size=512, 
               num_heads=8, 
               filter_size=2048, 
               dropout_rate=0.1,
               extra_decode_length=50,
               beam_width=4,
               alpha=0.6):
    """Constructor.

    Args:
      vocab_size: int scalar, num of subword tokens (including SOS/PAD and EOS) 
        in the vocabulary. 
      encoder_stack_size: int scalar, num of layers in encoder stack.
      decoder_stack_size: int scalar, num of layers in decoder stack.
      hidden_size: int scalar, the hidden size of continuous representation. 
      num_heads: int scalar, num of attention heads.
      filter_size: int scalar, the depth of the intermediate dense layer of the
        feed-forward sublayer.
      dropout_rate: float scalar, dropout rate for the Dropout layers.
      extra_decode_length: int scalar, the max decode length would be the sum of
        `tgt_seq_len` and `extra_decode_length`.
      beam_width: int scalar, beam width for beam search.
      alpha: float scalar, the parameter for length normalization used in beam 
        search.
    """
    super(TransformerModel, self).__init__()
    self._vocab_size = vocab_size
    self._encoder_stack_size = encoder_stack_size
    self._decoder_stack_size = decoder_stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
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

  def call(self, src_token_ids, tgt_token_ids):
    """Takes as input the source and target token ids, and returns the estimated
    logits for the target sequences. Note this function should be called in 
    training mode only.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.
      tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len], token ids 
        of target sequences.

    Returns:
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size]. 
    """
    padding_mask = utils.get_padding_mask(src_token_ids, SOS_ID)
    encoder_outputs = self._encode(src_token_ids, padding_mask, training=True)
    logits = self._decode(
        tgt_token_ids, encoder_outputs, padding_mask)
    return logits

  def _encode(self, src_token_ids, padding_mask, training=False):
    """Converts source sequences token ids into continuous representation, and 
    computes the Encoder-encoded sequences.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked). 
      training: bool scalar, True if in training mode.

    Returns:
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size], the encoded source sequences to be used as reference. 
    """
    src_seq_len = tf.shape(src_token_ids)[1]

    # [batch_size, src_seq_len, hidden_size]
    src_token_embeddings = self._embedding_logits_layer(
        src_token_ids, 'embedding')

    # [src_seq_len, hidden_size]
    positional_encoding = utils.get_positional_encoding(
        src_seq_len, self._hidden_size)
    src_token_embeddings += positional_encoding
    src_token_embeddings = self._encoder_dropout_layer(
        src_token_embeddings, training)

    encoder_outputs = self._encoder(
        src_token_embeddings, padding_mask, training)
    return encoder_outputs

  def _decode(self, tgt_token_ids, encoder_outputs, padding_mask):
    """Computes the estimated logits of target token ids, based on the encoded 
    source sequences. Note this function should be called in training mode only.

    Args:
      tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len] token ids of 
        target sequences.
      encoder_outputs: float tensor of shape [batch_size, src_seq_len, 
        hidden_size], the encoded source sequences to be used as reference. 
      padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len], 
        populated with either 0 (for tokens to keep) or 1 (for tokens to be 
        masked). 

    Returns:
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size].
    """
    tgt_seq_len = tf.shape(tgt_token_ids)[1]

    # [batch_size, tgt_seq_len, hidden_size]
    tgt_token_embeddings = self._embedding_logits_layer(
        tgt_token_ids, 'embedding')

    # [tgt_seq_len, hidden_size]
    positional_encoding = utils.get_positional_encoding(
        tgt_seq_len, self._hidden_size)
    tgt_token_embeddings += positional_encoding
    tgt_token_embeddings = self._decoder_dropout_layer(
        tgt_token_embeddings, training=True) 

    look_ahead_mask = utils.get_look_ahead_mask(tgt_seq_len)

    decoder_outputs = self._decoder(tgt_token_embeddings, 
                                    encoder_outputs, 
                                    look_ahead_mask, 
                                    padding_mask, 
                                    training=True)

    logits = self._embedding_logits_layer(decoder_outputs, 'logits')
    return logits

  def transduce(self, src_token_ids):
    """Takes as input the source token ids only, and outputs the token ids of 
    the decoded target sequences using beam search. Note this function should be 
    called in inference mode only.

    Args:
      src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.

    Returns:
      decoded_ids: int tensor of shape [batch_size, decoded_seq_len], the token
        ids of the decoded target sequences using beam search.
      scores: float tensor of shape [batch_size], the scores (length-normalized 
        log-probs) of the decoded target sequences.
      tgt_tgt_attention: a list of `decoder_stack_size` float tensor of shape 
        [batch_size, num_heads, decoded_seq_len, decoded_seq_len], 
        target-to-target attention weights.
      tgt_src_attention: a list of `decoder_stack_size` float tensor of shape 
        [batch_size, num_heads, decoded_seq_len, src_seq_len], target-to-source 
        attention weights.
      src_src_attention: a list of `encoder_stack_size` float tensor of shape 
        [batch_size, num_heads, src_seq_len, src_seq_len], source-to-source 
        attention weights.
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

    decoded_ids, scores, decoding_cache = bs.search(sos_ids, decoding_cache)

    tgt_tgt_attention = [
        decoding_cache['layer_%d' % i]['tgt_tgt_attention'].numpy()[:, 0]
        for i in range(self._decoder_stack_size)]
    tgt_src_attention = [
        decoding_cache['layer_%d' % i]['tgt_src_attention'].numpy()[:, 0]
        for i in range(self._decoder_stack_size)]

    decoded_ids = decoded_ids[:, 0, 1:]
    scores = scores[:, 0] 

    src_src_attention = [
        self._encoder._stack[i]._mha._attention_weights.numpy()
        for i in range(self._encoder._stack_size)]

    return (decoded_ids, scores, 
            tgt_tgt_attention, tgt_src_attention, src_src_attention)

  def _build_decoding_cache(self, src_token_ids, batch_size):
    """Builds a dictionary that caches previously computed key and value feature
    maps and attention weights of the growing decoded sequence.

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
          'v': tensor of shape [batch_size, 0, num_heads, size_per_head],
          'tgt_tgt_attention': tensor of shape [batch_size, num_heads, 
            0, 0],
          'tgt_src_attention': tensor of shape [batch_size, num_heads,
            0, src_seq_len].
    """
    padding_mask = utils.get_padding_mask(src_token_ids, SOS_ID)
    encoder_outputs = self._encode(src_token_ids, padding_mask, training=False)
    size_per_head = self._hidden_size // self._num_heads
    src_seq_len = padding_mask.shape[-1] 

    decoding_cache = {'layer_%d' % layer:
        {'k':
            tf.zeros([
                batch_size, 0, self._num_heads, size_per_head
            ], 'float32'),
         'v':
            tf.zeros([
                batch_size, 0, self._num_heads, size_per_head
            ], 'float32'),
         'tgt_tgt_attention':
            tf.zeros([
                batch_size, self._num_heads, 0, 0], 'float32'), 
         'tgt_src_attention':
            tf.zeros([
                batch_size, self._num_heads, 0, src_seq_len], 'float32')

        } for layer in range(self._decoder._stack_size)
    }
    decoding_cache['encoder_outputs'] = encoder_outputs
    decoding_cache['padding_mask'] = padding_mask
    return decoding_cache

  def _build_decoding_fn(self, max_decode_length):
    """Builds the decoding function that will be called in beam search.

    The function steps through the proposed token ids one at a time, and 
    generates the logits of next token id over the vocabulary.

    Args:
      max_decode_length: int scalar, the decoded sequences would not exceed
        `max_decode_length`.

    Returns:
      decoding_fn: a callable that outputs the logits of the next decoded token
        ids.
    """
    # [max_decode_length, hidden_size]
    timing_signal = utils.get_positional_encoding(
        max_decode_length, self._hidden_size)
    timing_signal = tf.cast(timing_signal, 'float32')

    def decoding_fn(decoder_input, cache, **kwargs):
      """Computes the logits of the next decoded token ids.

      Args:
        decoder_input: int tensor of shape [batch_size * beam_width, 1], the 
          decoded tokens at index `i`.
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
              size_per_head],
            'tgt_tgt_attention': tensor of shape [batch_size * beam_width, 
              num_heads, seq_len, seq_len],
            'tgt_src_attention': tensor of shape [batch_size * beam_width, 
              num_heads, seq_len, src_seq_len].
            Note `seq_len` is the running length of the growing decode sequence.
        kwargs: dict, storing the following additional keyword arguments.
          index -> int scalar tensor, the index of the `decoder_input` in the 
            decoded sequence.

      Returns:
        logits: float tensor of shape [batch_size * beam_width, vocab_size].
        cache: a dict with the same structure as the input `cache`, except that
          the shapes of the values of key `k`, `v`, `tgt_tgt_attention`, 
          `tgt_src_attention` are
          [batch_size * beam_width, seq_len + 1, num_heads, size_per_head],
          [batch_size * beam_width, seq_len + 1, num_heads, size_per_head],
          [batch_size * beam_width, num_heads, seq_len + 1, seq_len + 1],
          [batch_size * beam_width, num_heads, seq_len + 1, src_seq_len].
      """
      index = kwargs['index']
      # [batch_size * beam_width, 1, hidden_size]
      decoder_input = self._embedding_logits_layer(decoder_input, 'embedding')
      decoder_input += timing_signal[index:index + 1]

      decoder_outputs = self._decoder(decoder_input,
                                      cache['encoder_outputs'],
                                      tf.zeros((1, 1, 1, index + 1), 
                                          dtype='float32'),
                                      cache['padding_mask'],
                                      training=False,
                                      cache=cache)

      logits = self._embedding_logits_layer(decoder_outputs, mode='logits')
      logits = tf.squeeze(logits, axis=1)
      return logits, cache

    return decoding_fn 
