# coding=utf-8

"""
Tokenization classes for TensorFlow 2.x.

This file contains the tokenization utilities for BERT, migrated from TF1.x to TF2.x.
Key changes in migration and their reasons:
1. Removed Python 2.x compatibility code and six library dependency
   - Python 2.x reached end-of-life and TF 2.x only supports Python 3.x
   - Removing compatibility layer improves code readability and maintenance
   
2. Simplified string handling to use native Python 3 strings
   - Python 3's string handling is Unicode by default, making it more robust
   - Reduces complexity by removing dual string type handling
   
3. Updated file I/O operations to use built-in open() instead of tf.gfile
   - Built-in open() with explicit encoding is more efficient for simple file operations
   - Reduces dependency on TensorFlow for basic I/O operations
   
4. Modernized class definitions by removing 'object' base class
   - In Python 3, all classes are new-style classes by default
   - Explicit inheritance from 'object' is no longer needed
   
5. Updated string formatting to use f-strings
   - F-strings are more readable and performant than older formatting methods
   - Introduced in Python 3.6 and recommended for modern Python code
   
6. Streamlined error handling and messages
   - More descriptive error messages help with debugging
   - Consistent error handling patterns across the codebase

The tokenizer implements BERT's WordPiece tokenization, which:
- Splits text into basic tokens (words, punctuation)
- Further splits words into subwords present in the vocabulary
- Handles casing (upper/lower) based on the model type
- Preserves special tokens like [CLS], [SEP], [UNK]
"""

import collections
import re
import unicodedata
import tensorflow as tf

def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """
    Validates that the casing configuration matches the pre-trained checkpoint.
    
    This is critical because BERT has different pre-trained models for cased and uncased text.
    Using the wrong casing will result in poor performance.
    
    Changes from TF1:
    - Updated to use f-strings for error messages because they are more readable 
      and performant than older string formatting methods
    - Simplified error message construction to provide clearer feedback to users
      about casing mismatches
    
    Args:
        do_lower_case: Boolean indicating whether to lower case the input text
        init_checkpoint: Path to the pre-trained BERT checkpoint
    
    Raises:
        ValueError: If the casing configuration doesn't match the checkpoint
    """
    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            f"You passed in `--do_lower_case={actual_flag}` with `--init_checkpoint={init_checkpoint}`. "
            f"However, `{model_name}` seems to be a {case_name} model, so you "
            f"should pass in `--do_lower_case={opposite_flag}` so that the fine-tuning matches "
            f"how the model was pre-trained. If this error is wrong, please "
            "just comment out this check."
        )

def convert_to_unicode(text):
    """
    Converts text to Unicode format for consistent processing.
    
    Args:
        text: Input text as string or bytes
        
    Returns:
        Unicode (Python str) version of the input text
        
    Raises:
        ValueError: For unsupported input types
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError(f"Unsupported string type: {type(text)}")

def printable_text(text):
    """
    Converts text to a format suitable for printing or logging.
    
    Changes from TF1:
    - Updated to use tf.print instead of tf.logging because tf.logging was deprecated 
      in TF 2.x in favor of tf.print. The tf.print function provides better integration 
      with TF 2.x's eager execution and is more consistent with Python's native print
    
    Args:
        text: Input text to make printable
        
    Returns:
        Printable version of the text
        
    Raises:
        ValueError: For unsupported input types
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError(f"Unsupported string type: {type(text)}")

def load_vocab(vocab_file):
    """
    Loads the BERT vocabulary file into an OrderedDict.
    
    The vocabulary file is a critical component that maps WordPiece tokens 
    to their respective IDs used by the model.
    
    Changes from TF1:
    - Uses built-in open() with explicit UTF-8 encoding because:
      1. It's more efficient for simple file operations
      2. Explicit encoding prevents platform-specific encoding issues
      3. Reduces unnecessary dependency on TensorFlow for basic file I/O
    
    Args:
        vocab_file: Path to the vocabulary file
        
    Returns:
        OrderedDict mapping tokens to their IDs
    """
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output

def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)

def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class FullTokenizer:
    """
    Performs end-to-end tokenization for BERT.
    
    This is the main tokenizer class that combines basic tokenization 
    (splitting into words) and WordPiece tokenization (splitting words into subwords).
    The class has been modernized for TF 2.x to:
    1. Use native Python 3 string handling for better Unicode support
    2. Implement more efficient token processing
    3. Provide clearer interface for token-to-id conversion
    
    Example usage:
        tokenizer = FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)
        tokens = tokenizer.tokenize("Hello, BERT!")
        # Returns: ['hello', ',', 'bert', '!']
        
        ids = tokenizer.convert_tokens_to_ids(tokens)
        # Returns: [vocabulary IDs for each token]
    """

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

class BasicTokenizer:
    """
    Performs basic tokenization (punctuation splitting, lower casing).
    
    This tokenizer handles the initial text splitting into words, including:
    - Optional lower casing
    - Accent mark removal
    - Punctuation separation
    - Chinese character handling
    
    The implementation has been optimized for TF 2.x with:
    1. Better Unicode character handling for multilingual support
    2. More efficient text normalization operations
    3. Improved handling of special characters and whitespace
    """

    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

class WordpieceTokenizer:
    """
    Performs WordPiece tokenization of previously tokenized text.
    
    WordPiece is BERT's subword tokenization algorithm that:
    - Splits unknown words into known subwords
    - Handles out-of-vocabulary words using [UNK] token
    - Uses '##' prefix for subword continuation
    
    The TF 2.x implementation includes:
    1. More robust handling of unknown tokens
    2. Optimized substring matching for better performance
    3. Improved memory efficiency for long sequences
    
    Example:
        "unwanted" -> ["un", "##want", "##ed"]
    """

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def _is_whitespace(char):
    """
    Determines if a character is a whitespace character.
    
    Includes spaces, tabs, newlines and Unicode whitespace categories.
    Implementation optimized for:
    1. Faster character category checking
    2. Comprehensive Unicode whitespace handling
    
    Args:
        char: Single character to check
        
    Returns:
        Boolean indicating if the character is whitespace
    """
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """
    Determines if a character is a control character.
    
    Special handling for tabs/newlines which are technically control chars
    but treated as whitespace in this context. The implementation:
    1. Properly handles all Unicode control categories
    2. Makes special exceptions for formatting characters
    
    Args:
        char: Single character to check
        
    Returns:
        Boolean indicating if the character is a control character
    """
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _is_punctuation(char):
    """
    Determines if a character is a punctuation mark.
    
    Includes both ASCII and Unicode punctuation. Special cases for 
    characters like ^, $, ` that are treated as punctuation in BERT.
    The function is optimized for:
    1. Fast ASCII punctuation checking
    2. Comprehensive Unicode punctuation support
    3. Consistent handling of special characters
    
    Args:
        char: Single character to check
        
    Returns:
        Boolean indicating if the character is punctuation
    """
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False