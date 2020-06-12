"""Defines SubTokenizer class that encodes raw text into subword token ids, and
decodes the subword token ids back to the raw text. 
"""
import collections
import os
import re
import sys
import unicodedata

import tensorflow as tf

_ALPHANUMERIC_CHAR_SET = set(
    chr(i) for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith('L') or
       unicodedata.category(chr(i)).startswith('N'))
_ESCAPE_CHARS = set(u"\\_u;0123456789")

# vocab index of START-OF-SEQUENCE token (or PADDING token)
SOS_ID = 0
# vocab index of END-OF-SEQUENCE token
EOS_ID = 1
RESERVED_TOKENS = ['<pad>', '<EOS>']
_UNDEFINED_UNICODE = '\u3013'
_OOA_CHAR_REGEX = r'\\([0-9]+);'


class SubTokenizer(object):
  """Subtokenizer encodes raw text (a sequence of unicode chars) into a list of 
  subtoken IDs, or decodes a list of subtoken IDs back to the original raw text.

  Example:
    subtokenizer = Subtokenizer(subtoken_list, alphabet)

    # `text`: string
    # `subtoken_ids`: a list of ints
    subtoken_ids = subtokenizer.encode(text) 
    original_text = subtokenizer.decode(subtoken_ids)
    # `text` and `original_text` should be identical
 """
  def __init__(self, subtoken_list, alphabet, check_alphabet=True):
    """Constructor.

    Args:
      subtoken_list: a list of strings, the subword tokens.
      alphabet: a set, storing the unique unicode chars from all subtokens in 
        `subtoken_list`.
      check_alphabet: bool scalar, whether to check if the set of chars in 
        `subtoken_list` match the corresponding `alphabet`.
    """
    if check_alphabet:
      _check_alphabet_against_subtoken_list(alphabet, subtoken_list)

    self._subtoken_list = subtoken_list
    self._alphabet = alphabet 
    self._subtoken_dict = _subtoken_list_to_dict(subtoken_list)

    self._max_subtoken_length = 0
    for subtoken in subtoken_list:
      self._max_subtoken_length = max(self._max_subtoken_length, len(subtoken))

    self._token_seen = {}

  @property
  def vocab_size(self):
    return len(self._subtoken_list)

  def save_to_file(self, filename):
    """Save the subtokenizer's vocabulary to a '.subtoken' file and '.alphabet'
    file, from which the same subtokenizer can be restored later.
    
    Args:
      filename: string scalar, the name of the vocabulary file.
    """
    with tf.io.gfile.GFile(filename + '.subtokens', mode='w') as f:
      for subtoken in self._subtoken_list:
        f.write("'%s'\n" % subtoken)
    with tf.io.gfile.GFile(filename + '.alphabet', mode='w') as f:
      for char in self._alphabet:
        f.write("%s\n" % char)
 
  def encode(self, string, add_eos=False):
    """Encodes a raw string by

    1. Split it into a list of token strings.
    2. For each token string, split it into a list of subtoken strings.
    3. Translate subtoken strings into subtoken ids.

    Example:

    Args:
      string: a string scalar, the raw text to be encoded.
      add_eos: bool scalar, whether or not to add End-Of-Sequence `EOS` token.

    Returns:
      subtoken_ids: a list of ints, the subtoken ids.
    """
    subtoken_ids = [] 
    tokens = _split_string_to_tokens(string)
  
    for token in tokens:
      if token not in self._token_seen:
 
        subtokens = _split_token_to_subtokens(
            _escape_token(token, self._alphabet),  
            self._subtoken_dict, 
            self._max_subtoken_length)

        self._token_seen[token] = [
            self._subtoken_dict[subtoken] for subtoken in subtokens] 
      subtoken_ids.extend(self._token_seen[token]) 

    if add_eos:
      subtoken_ids.append(EOS_ID)
    return subtoken_ids


  def decode(self, subtoken_ids):
    """Decode a list of subtoken ids by

    1. Translate subtoken ids back to subtoken strings.
    2. Concatenate subtoken strings into a single string (separator is '').
    3. Split the single string by separator '_', which gives us the escaped 
        tokens.
    4. Unescape the escaped tokens, and join then with separator ' ' or ''.

    Example:

    Args:
      subtoken_ids: a list of ints, the subtoken ids
      
    Returns:
      a string scalar, the decoded string.
    """
    escaped_tokens = [self._subtoken_list[id_] for id_ in subtoken_ids 
        if id_ < self.vocab_size]

    escaped_tokens = ''.join(escaped_tokens)

    escaped_tokens = escaped_tokens.split('_')

    tokens = []
    for token in escaped_tokens:
      if token:
        tokens.append(_unescape_token(token))
    token_is_alnum = [token[0] in _ALPHANUMERIC_CHAR_SET for token in tokens] 

    string = []
    for i, token in enumerate(tokens):
      if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
        string.append(' ')
      string.append(token)
    return ''.join(string)


def restore_subtokenizer_from_vocab_files(filename):
  """Restores the subtokenizer from vocabulary files ('*.subtoken' and 
  '*.alphabet').

  Args:
    filename: string scalar, the name of the vocabulary file.

  Returns:
    subtokenizer: a SubTokenizer instance.
  """
  reserved_tokens = RESERVED_TOKENS

  subtoken_list = []
  alphabet = set()
  with tf.io.gfile.GFile(filename + '.subtokens', mode='r') as f:
    for line in f: 
      subtoken = line.strip()[1:-1] # drop leading and trailing single quotes
      if subtoken not in reserved_tokens:
        subtoken_list.append(subtoken)
  with tf.io.gfile.GFile(filename + '.alphabet', mode='r') as f:
    for line in f:
      char = line[:-1] # drop trailing '\n' 
      alphabet.add(char)

  subtoken_list = reserved_tokens + subtoken_list
  subtokenizer = SubTokenizer(subtoken_list, alphabet)
  return subtokenizer


def create_subtokenizer_from_raw_text_files(filenames, 
                                            target_vocab_size, 
                                            threshold, 
                                            min_count=None, 
                                            file_byte_limit=1e6):
  """Builds a vocabulary of subword tokens from raw text files and creates a 
  subtokenizer.

  If `min_count` is not None, build the vocabulary by calling 
  `_generate_subtokens` directly; Otherwise, use binary search to find the 
  `min_count` that results in the vocab size that is closest to 
  `target_vocab_size`.

  Args:
    filenames: a list of strings, names of raw text files. Line are separated
      by '\n', and each line is interpreted as a sentence.
    target_vocab_size: int scalar, the desired vocabulary size. Ignored if 
      `min_count` is not None.
    threshold: int scalar, if the difference between actual vocab size and 
      `target_vocab_size` is smaller than this, the binary search terminates. 
      Ignored if `min_count` is not None.
    min_count: int scalar, the minimum count required for a subtoken to be 
      included in the vocabulary. 
    file_byte_limit: int scalar, the max num of bytes worth of text to be 
      sampled from each raw text file to build the vocabulary.

  Returns:
    subtokenizer: a SubTokenizer instance.
  """ 
  # Build a dict mapping tokens (e.g. words in space-separated languages like 
  # English, or sentences in languages like Chinese, Japanese) to token counts. 
  token_counts = collections.defaultdict(int)

  for filepath in filenames:
    with tf.io.gfile.GFile(filepath, mode='r') as reader:
      # Sample approximately `file_byte_limit` bytes worth of text from each 
      # file. 
      file_byte_budget = file_byte_limit
      counter = 0
      lines_to_skip = int(reader.size() / (file_byte_budget * 1))
      for line in reader:
        if counter < lines_to_skip:
          counter += 1
        else:
          if file_byte_budget < 0:
            break
          line = line.strip()
          file_byte_budget -= len(line)
          counter = 0

          for token in _split_string_to_tokens(line):
            token_counts[token] += 1

  alphabet = _generate_alphabet(token_counts)

  if min_count is not None:
    subtoken_list = _generate_subtokens(token_counts, 
                                        alphabet, 
                                        min_count, 
                                        num_iterations=4) 
  else:
    subtoken_list = _generate_subtokens_with_target_vocab_size(
        token_counts, 
        alphabet, 
        target_vocab_size, 
        threshold)

  subtokenizer = SubTokenizer(subtoken_list, alphabet)
  return subtokenizer


def _check_alphabet_against_subtoken_list(alphabet, subtoken_list):
  derived_alphabet = {char for subtoken in subtoken_list for char in subtoken}
  if derived_alphabet != alphabet:
    raise ValueError('Alphabet derived from `subtoken_list` is different from '
        'input `alphabet`.')


def _split_string_to_tokens(text):
  """Split raw text string to tokens.

  The indices are first marked with a binary tag: whether or not the char at
  each index is in `_ALPHANUMERIC_CHAR_SET`. Then we break the string bewtween 
  index `pos - 1` and `pos`, if the tags at these two indices differ.

  Args:
    text: string scalar, raw text string of unicode chars.

  Returns:
    tokens: a list of strings, tokens from raw text string.
  """
  if len(text) == 0:
    return []
  tokens = []
  token_start = 0

  is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
  for pos in range(1, len(text)):
    if is_alnum[pos] != is_alnum[pos - 1]:
      token = text[token_start:pos]
      # if the resulting token is ' ', but it's not at the start of `text`,
      # we don't add it to the list `tokens`.
      if token != ' ' or token_start == 0:
        tokens.append(token)
      token_start = pos
  final_token = text[token_start:]
  tokens.append(final_token)
  return tokens


def _split_token_to_subtokens(token, subtoken_dict, max_subtoken_length):
  """Split a token string into a list of subtokens.

  Given a dictionary of valid subtokens and their maximum length, we greedily
  take the longest prefix of `token` that is also a valid subtoken, and add it 
  to the output list, and repeat the process on the remainder of `token`.

  Args:
    token: string scalar, the token to be split.
    subtoken_dict: a dict, with keys being all valid subtokens. 
    max_subtoken_length: string scalar, max length of subtokens in 
        `subtoken_dict`. 

  Returns:
    subtokens: a list of strings, subtokens split from `token`.
  """
  subtokens = []
  start = 0
  token_len = len(token)

  # in each iteration of while loop we take the longest prefix (valid subtoken)
  while start < token_len:
    # the prefix can't be longer than `max_subtoken_length`
    for end in range(min(token_len, start + max_subtoken_length), start, -1):
      subtoken = token[start:end]
      if subtoken in subtoken_dict:
        subtokens.append(subtoken)
        start = end
        break
    else:
      raise ValueError('Unable to split token "%s" into subtokens.' % token)
  return subtokens


def _subtoken_list_to_dict(subtoken_list):
  """Create dict mapping from subtokens to their ids.

  Args:
    subtoken_list: a list of str, subtokens.

  Returns:
    a dict mapping from subtoken string to its id.
  """
  return {subtoken: index for index, subtoken in enumerate(subtoken_list)}


def _escape_token(token, alphabet):
  """Escape a token string by performing the following operations.

  1. Replace '_' with '\\u'.
  2. Replace characters 'c' not in the alphabet with string `\ord(c);`.
  3. Append '_' to the resulting string.

  Args:
    token: string scalar, the token to be escaped.
    alphabet: a set, storing the unique unicode chars from tokens sampled from
      raw text files. 

  Returns:
    escaped_token: string scalar, the escaped token.
  """
  token = token.replace('_', '\\u')
  escaped_chars = [c if c in alphabet else r'\%d;' % ord(c) for c in token]
  escaped_token = ''.join(escaped_chars) + '_'
  return escaped_token


def _unescape_token(token):
  """Unescape the escaped token by performing the following operations.

  1. Replace '\\u' with '_'.
  2. Replace string `\ord(c);` with character 'c'.

  Note: the trailing '_' has already been dropped in the input string `token`.

  Args:
    token: string scalar, the escaped token.

  Returns:
    the unescaped token.
  """
  token = token.replace('\\u', '_')

  m = re.match(_OOA_CHAR_REGEX, token)
  if m is not None:
    try:
      return re.sub(_OOA_CHAR_REGEX, chr(int(m.group(1))), token)
    # if the unicode point `int(m.group(1))` is out of range, replace it with
    # `_UNDEFINED_UNICODE`
    except (ValueError, OverflowError) as _:
      return re.sub(_OOA_CHAR_REGEX, _UNDEFINED_UNICODE, token)
  return token


def _generate_subtokens(token_counts, alphabet, min_count, num_iterations=4):
  """Generates subtokens by breaking tokens into subtrings, and moving frequent
  ones into the final subtoken list.

  Args:
    token_counts: a dict, mapping token strings to their counts.
    alphabet: a set, storing the unique unicode chars from tokens sampled from
      raw text files.
    min_count: int scalar, the minimum count required for a subtoken to be 
      included in the vocabulary.
    num_iterations: int scalar, num of iterations. 

  Returns:
    subtoken_list: a list of strings, the subword tokens.
  """
  reserved_tokens = RESERVED_TOKENS

  # initial subtoken list: reserved tokens plus all chars in alphabet.
  # initial max subtoken length: 1 (because all chars have length 1)
  subtoken_list = reserved_tokens + list(alphabet)
  max_subtoken_length = 1

  for i in range(num_iterations):
    subtoken_dict = _subtoken_list_to_dict(subtoken_list)
 
    subtoken_counts = collections.defaultdict(int)
    for token, count in token_counts.items():
      token = _escape_token(token, alphabet)

      # greedily splits `token` into subtokens, giving priority to longer ones
      subtokens = _split_token_to_subtokens(
          token, subtoken_dict, max_subtoken_length)

      # adds additional subtokens:
      start = 0
      for subtoken in subtokens: 
        for end in range(start + 1, len(token) + 1):
          new_subtoken = token[start:end]
          subtoken_counts[new_subtoken] += count
        start += len(subtoken)

    subtoken_candidates = []

    # sort subtokens into buckets according to length
    # `subtoken_buckets[l]` stores the set of subtokens of length `l`
    subtoken_buckets = []
    for subtoken, count in subtoken_counts.items():
      if count < min_count:
        continue
      while len(subtoken_buckets) <= len(subtoken):
        subtoken_buckets.append(set()) 
      subtoken_buckets[len(subtoken)].add(subtoken)
    max_subtoken_length = len(subtoken_buckets) - 1


    for subtoken_len in range(max_subtoken_length, 0, -1):
      for subtoken in subtoken_buckets[subtoken_len]:
        count = subtoken_counts[subtoken]
        if count < min_count:
          continue

        # leave out subtokens in `alphabet` and `reserved_tokens`, which
        # will be added manually
        if subtoken not in alphabet and subtoken not in reserved_tokens:
          subtoken_candidates.append((count, subtoken))

        # if a subtoken is alread added to the candidate list, we remove all its
        # prefixes
        for end in range(1, subtoken_len):
          subtoken_counts[subtoken[:end]] -= count

    subtoken_candidates.extend((subtoken_counts[a], a) for a in alphabet) 
    subtoken_list = [t for _, t in sorted(subtoken_candidates, reverse=True)]
    subtoken_list = reserved_tokens + subtoken_list

  return subtoken_list   


def _generate_subtokens_with_target_vocab_size(
    token_counts, alphabet, target_vocab_size, threshold):
  """Use binary search to find the `min_count` that results in the vocab size 
  that is closest to `target_vocab_size`.

  Args:
    token_counts: a dict, mapping token strings to their counts.
    alphabet: a set, storing the unique unicode chars from tokens sampled from
      raw text files.
    target_vocab_size: int scalar, the desired vocabulary size. Ignored if 
      `min_count` is not None.
    threshold: int scalar, if the difference between actual vocab size and 
      `target_vocab_size` is smaller than this, the binary search terminates. 
      Ignored if `min_count` is not None.
  """
  def binary_search(min_val, max_val):
    cur_val = (min_val + max_val) // 2
    subtoken_list = _generate_subtokens(token_counts, 
                                        alphabet, 
                                        min_count=cur_val)

    val = len(subtoken_list)
    within_threshold = abs(val - target_vocab_size) < threshold

    # terminate if the different in with in `threshold`
    if within_threshold or min_val >= max_val or cur_val < 2:
      return subtoken_list
    elif val > target_vocab_size:
      other_subtoken_list = binary_search(cur_val + 1, max_val)
    else:
      other_subtoken_list = binary_search(min_val, cur_val - 1)

    # compare the vocab (i.e. `subtoken_list`) resulted from `cur_val` as the 
    # `min_count` or the vocab from either branch of the binary search (i.e. 
    # `other_subtoken_list`)
    other_val = len(other_subtoken_list)
    if abs(other_val - target_vocab_size) < abs(val - target_vocab_size):
      return other_subtoken_list
    else:
      return subtoken_list

  return binary_search(1, 1000)    


def _generate_alphabet(iterable):
  """Generates the alphabet (unique unicode chars) from an iterable of tokens.

  Note: the alphabet is augmented with chars from tokens in `RESERVED_TOKENS`
  and chars from `_ESCAPE_CHARS`: "<pad>EOS\\_u;0123456789"
  
  Arsg:
    iterable: an iterable (e.g. dict, list) of strings, the tokens.

  Returns:
    alphabet: a set, storing the unique unicode chars.
  """
  reserved_tokens = RESERVED_TOKENS

  alphabet = {c for token in iterable for c in token}
  alphabet |= {c for token in reserved_tokens for c in token}
  alphabet |= _ESCAPE_CHARS
  return alphabet


class BleuTokenizer(object):
  """Split raw text string (either groundtruth reference text or translated 
  text) into tokens to compute BLEU score.
  """
  def __init__(self):
    """Constructor."""
    # all punctuation characters
    punctuation = self._property_chars("P")
    # nondigit char followed by any punctuation char
    self._nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    # any punctuation char followed by nondigit char
    self._punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    # any symbol char
    self._symbol_re = re.compile("([" + self._property_chars("S") + "])")

  def _property_chars(self, prefix):
    """Returns all unicode chars whose category string starts with given prefix.

    Args:
      prefix: string scalar, the prefix that the unichode char's category string
        must start with.

    Returns:
      string scalar, the concatenation of all chars whose category string start
        with given prefix. 
    """
    return "".join(chr(x) for x in range(sys.maxunicode)
                   if unicodedata.category(chr(x)).startswith(prefix))

  def tokenize(self, string):
    """Tokenize a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/'
             'blob/master/scripts/generic/mteval-v14.pl#L954-L983

    Insert spaces upon seeing 
    1. [nondigit][punc]
    2. [punc][nondiigt]
    3. [symbol]
    
    Args:
      string: string scalar, the string to be tokenized. 

    Returns:
      a list of token strings.
    """
    string = self._nondigit_punct_re.sub(r"\1 \2 ", string)
    string = self._punct_nondigit_re.sub(r" \1 \2", string)
    string = self._symbol_re.sub(r" \1 ", string)
    return string.split()
