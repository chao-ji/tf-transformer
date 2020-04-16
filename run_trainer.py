"""Pipeline for training a Transformer model for neural machine translation. 
"""
import glob
import os

import tensorflow as tf
from absl import app
from absl import flags

import utils
from data import dataset
from data import tokenization
from model import TransformerModel
from model_runners import TransformerTrainer


SUFFIX = '*.tfrecord'

flags.DEFINE_string(
    'data_dir', None, 'Path to the directory storing all TFRecord files (with '
        'pattern *train*) for training.')
flags.DEFINE_string(
    'vocab_path', None, 'Path to the vocabulary file.')
flags.DEFINE_string(
    'model_dir', None, 'Path to the directory that checkpoint files will be '
        'written to.')

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
    'max_num_tokens', 4096, 'The maximum num of tokens in each batch.')
flags.DEFINE_integer(
    'max_length', 64, 'Source or target seqs longer than this will be filtered'
        ' out.')
flags.DEFINE_integer(
    'num_parallel_calls', 8, 'Num of TFRecord files to be processed '
        'concurrently.')

flags.DEFINE_float(
    'learning_rate', 2.0, 'Base learning rate.')
flags.DEFINE_float(
    'learning_rate_warmup_steps', 16000, 'Number of warm-ups steps.')
flags.DEFINE_float(
    'optimizer_adam_beta1', 0.9, '`beta1` of Adam optimizer.')
flags.DEFINE_float(
    'optimizer_adam_beta2', 0.997, '`beta2` of Adam optimizer.')
flags.DEFINE_float(
    'optimizer_adam_epsilon', 1e-9, '`epsilon` of Adam optimizer.')

flags.DEFINE_float(
    'label_smoothing', 0.1, 'Amount of probability mass withheld for negative '
        'classes.')
flags.DEFINE_integer(
    'num_steps', 100000, 'Num of training iterations (minibatches).')
flags.DEFINE_integer(
    'save_ckpt_per_steps', 5000, 'Every this num of steps to save checkpoint.')


FLAGS = flags.FLAGS

def main(_):  
  data_dir = FLAGS.data_dir
  vocab_path = FLAGS.vocab_path
  model_dir = FLAGS.model_dir

  encoder_stack_size = FLAGS.encoder_stack_size
  decoder_stack_size = FLAGS.decoder_stack_size
  hidden_size = FLAGS.hidden_size
  num_heads = FLAGS.num_heads
  filter_size = FLAGS.filter_size
  dropout_rate = FLAGS.dropout_rate

  max_num_tokens = FLAGS.max_num_tokens
  max_length = FLAGS.max_length
  num_parallel_calls = FLAGS.num_parallel_calls

  learning_rate = FLAGS.learning_rate
  learning_rate_warmup_steps = FLAGS.learning_rate_warmup_steps
  optimizer_adam_beta1 = FLAGS.optimizer_adam_beta1
  optimizer_adam_beta2 = FLAGS.optimizer_adam_beta2
  optimizer_adam_epsilon = FLAGS.optimizer_adam_epsilon 

  label_smoothing = FLAGS.label_smoothing
  num_steps = FLAGS.num_steps
  save_ckpt_per_steps = FLAGS.save_ckpt_per_steps

  # transformer model
  subtokenizer = tokenization.restore_subtokenizer_from_vocab_files(vocab_path)
  vocab_size = subtokenizer.vocab_size 
  model = TransformerModel(encoder_stack_size=encoder_stack_size,
                           decoder_stack_size=decoder_stack_size,
                           hidden_size=hidden_size, 
                           num_heads=num_heads,
                           filter_size=filter_size,
                           vocab_size=vocab_size,
                           dropout_rate=dropout_rate)

  # training dataset
  builder = dataset.TransformerDatasetBuilder(
      max_num_tokens, True, max_length, num_parallel_calls)
  filenames = sorted(glob.glob(os.path.join(data_dir, SUFFIX)))
  train_ds = builder.build_dataset(filenames)
  
  # learning rate and optimizer
  optimizer = tf.keras.optimizers.Adam(
      utils.LearningRateSchedule(learning_rate,
                                 hidden_size,
                                 learning_rate_warmup_steps),
      optimizer_adam_beta1, 
      optimizer_adam_beta2, 
      epsilon=optimizer_adam_epsilon)

  # checkpoint
  ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

  # build trainer and start training
  trainer = TransformerTrainer(model, label_smoothing)
  trainer.train(
      train_ds, optimizer, ckpt, model_dir, num_steps, save_ckpt_per_steps)


if __name__  == '__main__':
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('vocab_path')
  app.run(main)
