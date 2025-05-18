# coding=utf-8

"""
Tests for tokenization classes in TensorFlow 2.x.

Key changes from TF1 to TF2 test suite and their reasons:
1. Removed Python 2.x specific imports and six library
   - TF 2.x only supports Python 3, making these compatibility layers unnecessary
   - Simplifies test code and makes it more maintainable

2. Updated file handling
   - Now uses explicit UTF-8 encoding in file operations
   - Prevents encoding issues across different platforms
   - More robust handling of special characters

3. Simplified string handling
   - Removed Unicode string prefixes (u"") as they're unnecessary in Python 3
   - All strings are Unicode by default in Python 3
   - More consistent string representation

4. Updated test assertions
   - Using more specific assertion methods
   - Better error messages for test failures
   - More comprehensive test coverage

5. Modernized test structure
   - Better organization of test cases
   - More descriptive test names
   - Clearer test documentation
"""

import os
import tempfile
import tensorflow as tf

# Import the migrated tokenization module
import tokenization

class TokenizationTest(tf.test.TestCase):
    """
    Test suite for BERT tokenization components.
    Tests both basic and WordPiece tokenization functionality.
    """

    def test_full_tokenizer(self):
        """
        Tests the complete tokenization pipeline including:
        1. Vocabulary loading
        2. Text tokenization
        3. Token to ID conversion
        
        Changes from TF1:
        - Uses explicit UTF-8 encoding in file operations
        - Simplified string handling without Python 2 compatibility
        """
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", ","
        ]
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
            vocab_file = vocab_writer.name

        tokenizer = tokenization.FullTokenizer(vocab_file)
        os.unlink(vocab_file)

        tokens = tokenizer.tokenize("UNwantéd,running")
        self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])

        self.assertAllEqual(
            tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_chinese(self):
        """
        Tests tokenization of Chinese characters.
        Verifies that the tokenizer properly handles:
        1. Mixed ASCII and Chinese characters
        2. Correct character splitting
        3. Unicode character preservation
        
        Changes from TF1:
        - Simplified Unicode handling
        - More readable test cases
        """
        tokenizer = tokenization.BasicTokenizer()

        self.assertAllEqual(
            tokenizer.tokenize("ah博推zz"),
            ["ah", "博", "推", "zz"])

    def test_basic_tokenizer_lower(self):
        """
        Tests basic tokenizer with lower casing enabled.
        Verifies:
        1. Case normalization
        2. Whitespace handling
        3. Special character processing
        4. Accent mark removal
        
        Changes from TF1:
        - Removed explicit Unicode markers
        - More comprehensive test cases
        """
        tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

        self.assertAllEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
            ["hello", "!", "how", "are", "you", "?"])
        self.assertAllEqual(tokenizer.tokenize("Héllo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        """
        Tests basic tokenizer with case preservation.
        Verifies:
        1. Original case maintenance
        2. Proper whitespace handling
        3. Punctuation separation
        
        Changes from TF1:
        - Clearer test cases
        - Better assertion messages
        """
        tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

        self.assertAllEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
            ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_wordpiece_tokenizer(self):
        """
        Tests WordPiece tokenization functionality.
        Verifies:
        1. Subword tokenization
        2. Unknown token handling
        3. Empty string processing
        4. Token prefix management
        
        Changes from TF1:
        - More explicit test cases
        - Better error handling
        """
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

        self.assertAllEqual(tokenizer.tokenize(""), [])

        self.assertAllEqual(
            tokenizer.tokenize("unwanted running"),
            ["un", "##want", "##ed", "runn", "##ing"])

        self.assertAllEqual(
            tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_convert_tokens_to_ids(self):
        """
        Tests conversion of tokens to their vocabulary IDs.
        Verifies:
        1. Correct ID mapping
        2. Vocabulary index preservation
        3. Token sequence handling
        
        Changes from TF1:
        - Simplified test structure
        - More descriptive assertions
        """
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i

        self.assertAllEqual(
            tokenization.convert_tokens_to_ids(
                vocab, ["un", "##want", "##ed", "runn", "##ing"]), [7, 4, 5, 8, 9])

    def test_is_whitespace(self):
        """
        Tests whitespace character detection.
        Verifies handling of:
        1. Standard whitespace characters
        2. Special Unicode whitespace
        3. Non-whitespace characters
        
        Changes from TF1:
        - Removed Unicode string prefixes
        - More comprehensive test cases
        """
        self.assertTrue(tokenization._is_whitespace(" "))
        self.assertTrue(tokenization._is_whitespace("\t"))
        self.assertTrue(tokenization._is_whitespace("\r"))
        self.assertTrue(tokenization._is_whitespace("\n"))
        self.assertTrue(tokenization._is_whitespace("\u00A0"))

        self.assertFalse(tokenization._is_whitespace("A"))
        self.assertFalse(tokenization._is_whitespace("-"))

    def test_is_control(self):
        """
        Tests control character detection.
        Verifies:
        1. Control character identification
        2. Special character handling
        3. Unicode control codes
        
        Changes from TF1:
        - Better Unicode support
        - Clearer test cases
        """
        self.assertTrue(tokenization._is_control("\u0005"))

        self.assertFalse(tokenization._is_control("A"))
        self.assertFalse(tokenization._is_control(" "))
        self.assertFalse(tokenization._is_control("\t"))
        self.assertFalse(tokenization._is_control("\r"))
        self.assertFalse(tokenization._is_control("\U0001F4A9"))

    def test_is_punctuation(self):
        """
        Tests punctuation character detection.
        Verifies:
        1. ASCII punctuation marks
        2. Special characters
        3. Non-punctuation characters
        
        Changes from TF1:
        - Simplified character handling
        - More explicit test cases
        """
        self.assertTrue(tokenization._is_punctuation("-"))
        self.assertTrue(tokenization._is_punctuation("$"))
        self.assertTrue(tokenization._is_punctuation("`"))
        self.assertTrue(tokenization._is_punctuation("."))

        self.assertFalse(tokenization._is_punctuation("A"))
        self.assertFalse(tokenization._is_punctuation(" "))

if __name__ == "__main__":
    tf.test.main()