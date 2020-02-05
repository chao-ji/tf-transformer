"""Convert parallel corpus (raw text files) into TFRecord files."""
import itertools
import os

import tensorflow as tf
from absl import app
from absl import flags

import tokenization


FLAGS = flags.FLAGS

flags.DEFINE_list(
    'source_filenames', None, 'Names of files storing source language '
        'sequences.')
flags.DEFINE_list(
    'target_filenames', None, 'Names of files storing target language '
        'sequences.')
flags.DEFINE_float(
    'file_byte_limit', 1e6, 'Number of bytes to read from each text file.')
flags.DEFINE_integer(
    'target_vocab_size', 32768, 'The desired vocabulary size. Ignored if ' 
        '`min_count` is not None.')
flags.DEFINE_integer(
    'threshold', 327, 'If the difference between actual vocab size and '
        '`target_vocab_size` is smaller than this, the binary search '
        'terminates. Ignored if `min_count` is not None.')
flags.DEFINE_integer(
    'min_count', 6, 'The minimum count required for a subtoken to be ' 
        'included in the vocabulary.')
flags.DEFINE_string(
    'vocab_name', 'vocab', 'Vocabulary will be stored in two files: '
        '"vocab.subtokens", "vocab.alphabet".')
flags.DEFINE_integer(
    'total_shards', 100, 'Total number of shards of the dataset (number of the '
        'generated TFRecord files)')
flags.DEFINE_string(
    'output_dir', None, 'Path to the directory that the generated TFRecord '
        'files will be written to')


def main(_):
  source_filenames = FLAGS.source_filenames
  target_filenames = FLAGS.target_filenames
  file_byte_limit = FLAGS.file_byte_limit
  target_vocab_size = FLAGS.target_vocab_size
  threshold = FLAGS.threshold
  min_count = FLAGS.min_count
  vocab_name = FLAGS.vocab_name
  output_dir = FLAGS.output_dir
  total_shards = FLAGS.total_shards
  train_files_flat = source_filenames + target_filenames

  subtokenizer = tokenization.create_subtokenizer_from_raw_text_files(
      train_files_flat, 
      target_vocab_size, 
      threshold, 
      min_count=min_count, 
      file_byte_limit=file_byte_limit)

  subtokenizer.save_to_file(vocab_name)

  source_files = [tf.io.gfile.GFile(fn) for fn in source_filenames]
  target_files = [tf.io.gfile.GFile(fn) for fn in target_filenames]

  source_data = itertools.chain(*source_files)
  target_data = itertools.chain(*target_files)

  filepaths = [os.path.join(output_dir, '%05d-of-%05d.tfrecord' % 
      (i + 1, total_shards))  for i in range(total_shards)]

  writers = [tf.io.TFRecordWriter(fn) for fn in filepaths]
  shard = 0

  for counter, (source_line, target_line) in enumerate(zip(
      source_data, target_data)):
    source_line = source_line.strip()
    target_line = target_line.strip()
    if counter > 0 and counter % 1e5 == 0:
        print('Number of examples saved: %d.' % counter)

    example = _dict_to_example(
        {"source": subtokenizer.encode(source_line, add_eos=True),
         "target": subtokenizer.encode(target_line, add_eos=True)})
    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % total_shards

  for writer in writers:
    writer.close()


def _dict_to_example(dictionary):
  features = {}
  for k, v in dictionary.items():
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


if __name__ == '__main__':
  flags.mark_flag_as_required('source_filenames')
  flags.mark_flag_as_required('target_filenames')
  flags.mark_flag_as_required('output_dir')

  app.run(main)
