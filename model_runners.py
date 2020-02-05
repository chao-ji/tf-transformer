"""Defines Trainer, Evaluator and Inferencer classes of Transformer model.
"""
import math
import os
import pickle

import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

import utils
from data import tokenization


class TransformerTrainer(object):
  """Trains a Transformer model."""
  def __init__(self, model, label_smoothing):
    """Constructor.

    Args:
      model: an instance of TransformerModel instance.
      label_smoothing: float scalar, applies label smoothing to the one-hot 
        class labels. Positive class has prob mass 1 - `label_smoothing`, while 
        each negative class has prob mass `label_smoothing / num_neg_classes`.
    """
    self._model = model
    self._label_smoothing = label_smoothing

  def train(self, 
            dataset,
            optimizer,
            ckpt,
            ckpt_path,
            num_iterations,
            persist_per_iterations,
            log_per_iterations=100,
            logdir='log'):
    """Performs training iterations.

    Args:
      dataset: a tf.data.Dataset instance, the input data generator.
      optimizer: a tf.keras.optimizer.Optimizer instance, applies gradient 
        updates.
      ckpt: a tf.train.Checkpoint instance, saves or load weights to/from 
        checkpoint file.
      ckpt_path: string scalar, the path to the directory that the checkpoint 
        files will be written to or loaded from.
      num_iterations: int scalar, num of iterations to train the model.
      persist_per_iterations: int scalar, saves weights to checkpoint files
        every `persist_per_iterations` iterations.
      log_per_iterations: int scalar, prints log info every `log_per_iterations`
        iterations.
      logdir: string scalar, the directory that the tensorboard log data will
        be written to. 
    """ 
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]

    @tf.function(input_signature=train_step_signature)
    def train_step(src_token_ids, tgt_token_ids):
      with tf.GradientTape() as tape:
        # pad `tgt_token_ids` with zeros as `SOS_ID`s
        tgt_token_ids_input = tf.pad(tgt_token_ids, [[0, 0], [1, 0]])[:, :-1]
        logits = self._model(src_token_ids, tgt_token_ids_input, training=True)
        loss = utils.compute_loss(tgt_token_ids, 
                                  logits, 
                                  self._label_smoothing, 
                                  self._model._vocab_size)

      gradients = tape.gradient(loss, self._model.trainable_variables)
      optimizer.apply_gradients(
          zip(gradients, self._model.trainable_variables))

      step = optimizer.iterations
      lr = optimizer.learning_rate(step)
      return loss, step - 1, lr

    summary_writer = tf.summary.create_file_writer(logdir)

    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
      print('Restoring from checkpoint: %s ...' % latest_ckpt)
      ckpt.restore(latest_ckpt)
    else:
      print('Training from scratch...')

    for src_token_ids, tgt_token_ids in dataset:
      loss, step, lr = train_step(src_token_ids, tgt_token_ids)

      with summary_writer.as_default():
        tf.summary.scalar('train_loss', loss, step=step)
        tf.summary.scalar('learning_rate', lr, step=step)

      if step.numpy() % log_per_iterations == 0:
        print('global step: %d, loss: %f, learning rate:' % 
            (step.numpy(), loss.numpy()), lr.numpy())
      if step.numpy() % persist_per_iterations == 0:
        print('Saving checkpoint at global step %d ...' % step.numpy())
        ckpt.save(os.path.join(ckpt_path, 'transformer'))

      if step.numpy() == num_iterations:
        break


class TransformerEvaluator(object):
  """Evaluates a Transformer model."""
  def __init__(self, model, subtokenizer, decode_batch_size, decode_max_length):
    """Constructor.

    Args:
      model: an instance of TransformerModel instance.
      subtokenizer: a SubTokenizer instance.
      decode_batch_size: int scalar, num of sequences in a batch to be decoded.
      decode_max_length: int scalar, max length of decoded sequence.
    """
    self._model = model
    self._decode_batch_size = decode_batch_size
    self._subtokenizer = subtokenizer
    self._decode_max_length = decode_max_length

    self._bleu_tokenizer = tokenization.BleuTokenizer()

  def translate(self,
                source_text_filename,
                output_filename=None):
    """Translate the source sequences.

    Args:
      source_text_filename: string scalar, name of the text file storing source 
        sequences, lines separated by '\n'.
      output_filename: (Optional) name of the file that translations will be 
        written to.
    """
    sorted_lines, sorted_indices = _get_sorted_lines(source_text_filename)

    total_samples = len(sorted_lines)
    batch_size = self._decode_batch_size
    num_decode_batches = math.ceil(total_samples / batch_size)

    def input_generator():
      # generate batched source sequences of token ids of shape 
      # [batch_size, decode_max_length]
      for i in range(num_decode_batches):
        lines = [sorted_lines[j + i * batch_size]
            for j in range(batch_size)
            if j + i * batch_size < total_samples]
        lines = [self._subtokenizer.encode(l, add_eos=True) for l in lines]
        batch = tf.keras.preprocessing.sequence.pad_sequences(
            lines,
            maxlen=self._decode_max_length,
            dtype='int32',
            padding='post')
        yield batch

    translations = []
    for i, source_ids in enumerate(input_generator()):
      # transduce `source_ids` into `translated_ids`, trim token ids at and 
      # beyond EOS, and decode ids into text
      translated_ids, _ = self._model.transduce(source_ids)
      translated_ids = translated_ids.numpy()
      length = len(translated_ids)

      for j in range(length):
        if j + i * batch_size < total_samples:
          translation = self._trim_and_decode(translated_ids[j])
          translations.append(translation)

    # optionally write translations to a text file
    if output_filename is not None:
      _write_translations(output_filename, sorted_indices, translations)
    return translations, sorted_indices

  def evaluate(self,
               source_text_filename, 
               target_text_filename=None, 
               output_filename=None):
    """Translate the source sequences and compute the BLEU score of the 
    translations against groundtruth target sequences.

    Args:
      source_text_filename: string scalar, name of the text file storing source 
        sequences, lines separated by '\n'.
      target_text_filename: string scalar, name of the text file storing target
        sequences, lines separated by '\n'.
      output_filename: (Optional) name of the file that translations will be 
        written to.
    """
    translations, sorted_indices = self.translate(
        source_text_filename, output_filename)

    if target_text_filename is None:
      return None, None

    targets = tf.io.gfile.GFile(
        target_text_filename).read().strip().splitlines()
    translations = [translations[i] for i in sorted_indices]

    # compute BLEU score if case-sensitive
    targets_tokens = [self._bleu_tokenizer.tokenize(x) for x in targets]
    translations_tokens = [self._bleu_tokenizer.tokenize(x) 
        for x in translations]
    case_sensitive_score = corpus_bleu(
        [[s] for s in targets_tokens], translations_tokens) * 100

    # compute BLEU score if case-insensitive (lower case)
    targets = [x.lower() for x in targets]
    translations = [x.lower() for x in translations]
    targets_tokens = [self._bleu_tokenizer.tokenize(x) for x in targets]
    translations_tokens = [self._bleu_tokenizer.tokenize(x) 
        for x in translations]
    case_insensitive_score = corpus_bleu(
        [[s] for s in targets_tokens], translations_tokens) * 100

    return case_insensitive_score, case_sensitive_score

  def _trim_and_decode(self, ids):
    """Trim tokens at EOS and beyond EOS in ids, and decode the remaining ids to
    text.

    Args:
      ids: numpy array of shape [num_ids], the translated ids ending with EOS id
        and padded with PAD ID. 

    Returns:
      string scalar, the decoded text string.
    """
    try:
      index = list(ids).index(tokenization.EOS_ID)
      return self._subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
      return self._subtokenizer.decode(ids)

def _get_sorted_lines(filename):
  """Read raw text lines from a text file, and sort the lines in descending 
  order of the num of space-separated words.

  Example:
    text file: (l1, 0), (l2, 1), (l3, 2), (l4, 3), (l5, 4)

    sorted: (l2, 1), (l5, 4), (l4, 3), (l1, 0), (l3, 2)

    sorted_lines: l2, l5, l4, l1, l3
    sorted_indices: 3, 0, 4, 2, 1

  Args:
    filename: string scalar, the name of the file that raw text will be read 
      from.

  Returns:
    sorted_lines: a list of strings, storing the raw text files sorted in 
      descending order of `num_words`.
    sorted_indices: a list of ints, `[sorted_lines[i] for i in sorted_indices]` 
      would restore the lines in the order as they appear in original text file.
  """
  with tf.io.gfile.GFile(filename) as f:
    lines = f.read().split('\n')
    lines = [line.strip() for line in lines]
    if len(lines[-1]) == 0:
      lines.pop()

  # each line is converted to tuple (index, num_of_words, raw_text), and 
  # sorted in descending order of `num_words`
  lines_w_lengths = [(i, len(line.split()), line)
      for i, line in enumerate(lines)]
  lines_w_lengths = sorted(lines_w_lengths, key=lambda l: l[1], reverse=True)
  
  sorted_indices = [None] * len(lines_w_lengths)
  sorted_lines = [None] * len(lines_w_lengths)

  # `index` is the index of `line` in the original unsorted file
  for i, (index, _, line) in enumerate(lines_w_lengths):
    sorted_indices[index] = i
    sorted_lines[i] = line

  return sorted_lines, sorted_indices

def _write_translations(output_filename, sorted_indices, translations):
  """Write translations to a text file.

  Args:
    output_filename: string scalar, name of the file that translations will be 
      written to.
    sorted_indices: a list of ints, `[translations[i] for i in sorted_indices]`
      would produce the translations of text lines in the original text file.
    translations: a list of strings, the tranlations. 
  """
  with tf.io.gfile.GFile(output_filename, "w") as f:
    for i in sorted_indices:
      f.write("%s\n" % translations[i])


class TransformerInferencer(object):
  def __init__(self):
    pass

