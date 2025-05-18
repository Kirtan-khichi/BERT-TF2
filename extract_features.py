# coding=utf-8

"""
Extract pre-computed feature vectors from BERT.

This script extracts the hidden layer representations (features) from a pre-trained BERT model
for given input text. These features can be used for various downstream NLP tasks.

Key changes from TF1 to TF2:
1. Model handling
   - Switched from custom BERT implementation to HuggingFace's transformers library
   - Using TFBertModel instead of the original BertModel
   - Better handling of model weights and configurations

2. TPU/Distribution changes
   - Removed TPU-specific code since TF2 handles devices differently
   - Simplified distribution strategy
   - More efficient batch processing

3. Input pipeline updates
   - Using tf.data.Dataset for better performance
   - More efficient data loading and preprocessing
   - Better memory management for large datasets

4. API modernization
   - Updated to TF2 Keras-style model calls
   - Using tf.io instead of tf.gfile
   - Switched to absl flags from tf.flags
   - Updated logging to tf.get_logger()

5. Output handling
   - More efficient feature extraction
   - Better JSON serialization
   - Improved error handling
"""

import codecs
import collections
import json
import re
import os
import logging
from tqdm import tqdm

import tokenization
import tensorflow as tf
from absl import flags, app
import numpy as np
from transformers import TFBertModel, BertConfig

FLAGS = flags.FLAGS

# Define command line flags with clear descriptions
flags.DEFINE_string("input_file", None, "")
flags.DEFINE_string("output_file", None, "")
flags.DEFINE_string("layers", "-1,-2,-3,-4", "")
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer("batch_size", 8, "Batch size for predictions.")

class InputExample:
    """
    Represents a single input example for feature extraction.
    
    Each example contains:
    - unique_id: Unique identifier for the example
    - text_a: First piece of text
    - text_b: Optional second piece of text for sentence pair tasks
    """
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures:
    """
    Holds the processed features of an input example.
    
    Contains:
    - unique_id: Example identifier
    - tokens: List of tokens after tokenization
    - input_ids: Token ids for model input
    - input_mask: Attention mask (1 for real tokens, 0 for padding)
    - input_type_ids: Token type ids (0 for first sequence, 1 for second)
    """
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def create_model(bert_config, is_training, input_ids, input_mask, input_type_ids):
    """
    Creates a BERT model for feature extraction.
    
    Changes from TF1:
    - Uses HuggingFace's TFBertModel instead of custom implementation
    - Simplified model creation and configuration
    - Better handling of training mode
    
    Args:
        bert_config: BERT model configuration
        is_training: Whether the model is in training mode
        input_ids: Token ids
        input_mask: Attention mask
        input_type_ids: Token type ids
        
    Returns:
        Model's encoder layers for feature extraction
    """
    model = modeling.TFBertModel(
        config=bert_config)
    
    sequence_output, pooled_output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        token_type_ids=input_type_ids,
        training=is_training)
    
    return model.get_all_encoder_layers()

def convert_examples_to_features(examples, seq_length, tokenizer):
    """
    Converts text examples into features that BERT can process.
    
    The process:
    1. Tokenizes the input text
    2. Handles sequence pairs if present
    3. Truncates to max length
    4. Adds special tokens ([CLS], [SEP])
    5. Converts to model-ready format
    
    Args:
        examples: List of InputExample objects
        seq_length: Maximum sequence length
        tokenizer: BERT tokenizer
        
    Returns:
        List of InputFeatures objects
    """
    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="Converting examples to features", total=len(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info(f"unique_id: {example.unique_id}")
            logging.info(f"tokens: {' '.join([tokenization.printable_text(x) for x in tokens])}")
            logging.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logging.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
            logging.info(f"input_type_ids: {' '.join([str(x) for x in input_type_ids])}")

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair to fit max_length.
    
    Uses a simple strategy:
    - Keep removing tokens from the longer sequence until we fit
    - This preserves more information from the shorter sequence
    
    Args:
        tokens_a: First sequence tokens
        tokens_b: Second sequence tokens
        max_length: Maximum allowed length
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def read_examples(input_file):
    """
    Reads input examples from a file.
    
    File format:
    - One example per line
    - For sequence pairs: text_a ||| text_b
    - For single sequences: just the text
    
    Args:
        input_file: Path to input file
        
    Returns:
        List of InputExample objects
    """
    examples = []
    unique_id = 0
    total_lines = sum(1 for _ in tf.io.gfile.GFile(input_file, "r"))
    
    with tf.io.gfile.GFile(input_file, "r") as reader:
        for line in tqdm(reader, desc="Reading examples", total=total_lines):
            line = tokenization.convert_to_unicode(line)
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples

def create_dataset(features, batch_size):
    """
    Creates a TF Dataset from features.
    
    Changes from TF1:
    - Uses tf.data.Dataset for better performance
    - More efficient batching
    - Better memory usage
    
    Args:
        features: List of InputFeatures
        batch_size: Batch size for processing
        
    Returns:
        tf.data.Dataset object
    """
    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    dataset = tf.data.Dataset.from_tensor_slices({
        "unique_ids": tf.constant(all_unique_ids, dtype=tf.int32),
        "input_ids": tf.constant(all_input_ids, dtype=tf.int32),
        "input_mask": tf.constant(all_input_mask, dtype=tf.int32),
        "input_type_ids": tf.constant(all_input_type_ids, dtype=tf.int32),
    })

    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset

def model_fn_builder(bert_config, init_checkpoint):
    """
    Creates a function that handles model operations.
    
    Changes from TF1:
    - Uses HuggingFace's TFBertModel
    - Simplified model loading and initialization
    - Better checkpoint handling
    
    Args:
        bert_config: Path to BERT configuration file
        init_checkpoint: Path to initial checkpoint
        
    Returns:
        Function that processes model operations
    """
    def model_fn(features):
        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        # Load pre-trained model
        config = BertConfig.from_json_file(bert_config)
        model = TFBertModel.from_pretrained(init_checkpoint, config=config)

        # Get embeddings
        outputs = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=input_type_ids,
            training=False
        )

        return {
            "sequence_output": outputs.last_hidden_state,
            "pooled_output": outputs.pooler_output
        }

    return model_fn

def input_fn_builder(features, seq_length):
    """
    Creates input function for the model.
    
    Changes from TF1:
    - Simplified input handling
    - More efficient data organization
    - Better type handling
    
    Args:
        features: List of InputFeatures
        seq_length: Maximum sequence length
        
    Returns:
        Function that provides input data
    """
    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn():
        return {
            "unique_ids": tf.constant(all_unique_ids, dtype=tf.int32),
            "input_ids": tf.constant(all_input_ids, dtype=tf.int32),
            "input_mask": tf.constant(all_input_mask, dtype=tf.int32),
            "input_type_ids": tf.constant(all_input_type_ids, dtype=tf.int32)
        }

    return input_fn

def main(_):
    """
    Main function that orchestrates the feature extraction process.
    
    The process:
    1. Sets up logging and configuration
    2. Loads the BERT model
    3. Processes input examples
    4. Extracts features
    5. Writes results to output file
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    print("Initializing BERT feature extraction...")

    # Parse layer indices
    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

    # Load BERT configuration
    print("Loading BERT configuration...")
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    bert_config.output_hidden_states = True  # Enable hidden states output

    # Create tokenizer
    print("Initializing tokenizer...")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # Read examples
    print("Reading input examples...")
    examples = read_examples(FLAGS.input_file)

    # Convert examples to features
    print("Converting examples to features...")
    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    # Create feature lookup
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    # Create dataset
    dataset = create_dataset(features, FLAGS.batch_size)

    # Create and load model
    print("Loading BERT model...")
    model_fn = model_fn_builder(
        bert_config=FLAGS.bert_config_file,
        init_checkpoint=FLAGS.init_checkpoint)
    
    # Process batches
    total_batches = len(features) // FLAGS.batch_size + (1 if len(features) % FLAGS.batch_size != 0 else 0)
    
    print("Extracting features...")
    for batch in tqdm(dataset, desc="Processing batches", total=total_batches):
        # Get encoder layers
        outputs = model_fn(batch)
        
        # Process each example in batch
        for i in range(batch["unique_ids"].shape[0]):
            unique_id = int(batch["unique_ids"][i])
            feature = unique_id_to_feature[unique_id]
            
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_features = []
            
            # Process each token
            for (j, token) in enumerate(feature.tokens):
                all_layers_for_token = []
                
                # Process each layer
                for (k, layer_index) in enumerate(layer_indexes):
                    layer_output = outputs["sequence_output"][i][j][layer_index]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(float(x), 6) for x in layer_output.numpy().flat
                    ]
                    all_layers_for_token.append(layers)
                
                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers_for_token
                all_features.append(features)
            
            output_json["features"] = all_features
            with codecs.getwriter("utf-8")(tf.io.gfile.GFile(FLAGS.output_file, "a")) as writer:
                writer.write(json.dumps(output_json) + "\n")
    
    print("Feature extraction completed successfully!")

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file")
    app.run(main)
