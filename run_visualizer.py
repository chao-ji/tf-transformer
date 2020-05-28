"""Visualizes the attention weights."""
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from absl import flags

from data import tokenization


flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_string(
    'attention_file', None, 'Path to the *.npy file storing attention weights.')
flags.DEFINE_integer(
    'index', 0, 'Index of the source-target sequence pair in a batch.')
flags.DEFINE_integer(
    'head', 0, 'Index of the attention heads.')

FLAGS = flags.FLAGS


def draw_attention_weights(query, 
                           reference, 
                           attention_weights, 
                           subtoken_list, 
                           figsize=(16, 16)):
  """Draws the source-source, target-source, and target-target attention 
  weights.

  Args:
    query: int numpy array of shape [query_seq_len], the list of subtoken ids 
      ending with EOS_ID, and maybe zero-padded.
    reference: int numpy array of shape [ref_seq_len], the list of subtoken ids
      ending with EOS_ID, and maybe zero-padded.
    attention_weights: float numpy array of shape[query_seq_len, ref_seq_len], 
      the attention weights.
    subtoken_list: a list of strings, the subword tokens listed in the order of
      their vocabulary indices.
    figsize: tuple of two ints, figure size. 
  """
  q_len = list(query).index(1) + 1
  r_len = list(reference).index(1) + 1
  attention_weights = attention_weights[:q_len, :r_len]

  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(111)
  mat = ax.matshow(attention_weights, cmap='viridis')

  ax.set_xticks(range(r_len))
  ax.set_yticks(range(q_len))

  ax.set_xticklabels(
    [subtoken_list[reference[i]] for i in range(r_len)], rotation=45)

  ax.set_yticklabels(
    [subtoken_list[query[i]] for i in range(q_len)])

  ax.set_xlabel('Reference sequence.')
  ax.set_ylabel('Query sequence.')
  ax.xaxis.set_label_position('top')
  fig.colorbar(mat)


def main(_):
  index = FLAGS.index
  head = FLAGS.head
  vocab_path = FLAGS.vocab_path
  attention_file = FLAGS.attention_file
 
  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  subtoken_list = subtokenizer._subtoken_list

  attn = np.load(attention_file, allow_pickle=True).item()

  src_src = attn['src_src_attention']
  tgt_src = attn['tgt_src_attention']
  tgt_tgt = attn['tgt_tgt_attention']
  src = attn['src']
  tgt = attn['tgt']

  draw_attention_weights(
    src[index], src[index], src_src[index, head], subtoken_list)
  plt.savefig('src_src.png', dpi=256)
  draw_attention_weights(
    tgt[index], src[index], tgt_src[index, head], subtoken_list)
  plt.savefig('tgt_src.png', dpi=256)
  draw_attention_weights(
    tgt[index], tgt[index], tgt_tgt[index, head], subtoken_list)
  plt.savefig('tgt_tgt.png', dpi=256)


if __name__ == '__main__':
  flags.mark_flag_as_required('vocab_path')
  flags.mark_flag_as_required('attention_file')
  app.run(main)
