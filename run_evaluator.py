"""Translate source sequences into target sequences, and optionally evaluates 
BLEU score in groundtruth target sequences are provided.
"""
import tensorflow as tf
from absl import app
from absl import flags

from data import tokenization
from model import TransformerModel
from model_runners import TransformerEvaluator


flags.DEFINE_integer(
    'encoder_stack_size', 6, 'Num of layers in encoder stack.')
flags.DEFINE_integer(
    'decoder_stack_size', 6, 'Num of layers in decoder stack.')
flags.DEFINE_integer(
    'hidden_size', 512, 'The dimensionality of the embedding vector.')
flags.DEFINE_integer(
    'num_heads', 8, 'Num of attention heads.')
flags.DEFINE_integer(
    'filter_size', 2048, 'The depth of the intermediate dense layer of the'
        'feed-forward sublayer.')
flags.DEFINE_float(
    'dropout_rate', 0.1, 'Dropout rate for the Dropout layers.')
flags.DEFINE_integer(
    'extra_decode_length', 50, 'The max decode length would be'
        ' the sum of `tgt_seq_len` and `extra_decode_length`.')
flags.DEFINE_integer(
    'beam_width', 4, 'Beam width for beam search.')
flags.DEFINE_float(
    'alpha', 0.6, 'The parameter for length normalization used in beam search.')
flags.DEFINE_integer(
    'decode_batch_size', 32, 'Number of sequences in a batch for inference.')
flags.DEFINE_integer(
    'decode_max_length', 97, 'Max number of tokens that will be decoded for a '
        'given source sequence.') 


flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_string(
    'source_text_filename', None, 'Path to the source text sequences to be '
        'translated.')
flags.DEFINE_string(
    'target_text_filename', None, 'Path to the target (reference) text '
        'sequences that the translation will be checked against,')
flags.DEFINE_string(
    'translation_output_filename', 'translations.text', 'Path to the output '
        'file that the translations will be written to.')
flags.DEFINE_string(
    'model_dir', None, 'Path to the directory that checkpoint files will be '
        'written to.')

FLAGS = flags.FLAGS


def main(_):
  vocab_path = FLAGS.vocab_path
  model_dir = FLAGS.model_dir

  encoder_stack_size = FLAGS.encoder_stack_size
  decoder_stack_size = FLAGS.decoder_stack_size
  hidden_size = FLAGS.hidden_size
  num_heads = FLAGS.num_heads
  filter_size = FLAGS.filter_size
  dropout_rate = FLAGS.dropout_rate

  extra_decode_length = FLAGS.extra_decode_length
  beam_width = FLAGS.beam_width
  alpha = FLAGS.alpha
  decode_batch_size = FLAGS.decode_batch_size
  decode_max_length = FLAGS.decode_max_length

  source_text_filename = FLAGS.source_text_filename
  target_text_filename = FLAGS.target_text_filename
  translation_output_filename = FLAGS.translation_output_filename


  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)

  vocab_size = 33708 # subtokenizer.vocab_size
  model = TransformerModel(encoder_stack_size=encoder_stack_size,
                           decoder_stack_size=decoder_stack_size,
                           hidden_size=hidden_size,
                           num_heads=num_heads,
                           filter_size=filter_size,
                           vocab_size=vocab_size,
                           dropout_rate=dropout_rate)

  ckpt = tf.train.Checkpoint(model=model)

  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  print(latest_ckpt, '\n\n\n')
  ckpt.restore(latest_ckpt).expect_partial()


  evaluator = TransformerEvaluator(
      model, subtokenizer, decode_batch_size, decode_max_length)


  case_insensitive_score, case_sensitive_score = evaluator.evaluate(
      source_text_filename, target_text_filename, translation_output_filename) 

  print(case_insensitive_score, case_sensitive_score)

if __name__ == '__main__':
  flags.mark_flag_as_required('source_text_filename')
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('vocab_path')
  app.run(main) 
