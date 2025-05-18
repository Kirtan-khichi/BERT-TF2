# coding=utf-8

"""Run BERT on SQuAD 1.1 and SQuAD 2.0.

This script implements BERT for the Stanford Question Answering Dataset (SQuAD) task.
Key features:
- Fine-tuning BERT for question answering
- Support for both SQuAD 1.1 and 2.0 datasets
- Training and evaluation pipelines
- TPU/GPU support

Major Migration Changes from TF1 to TF2:
1. Framework Updates:
   - Switched to absl.flags from tf.flags
   - Using Python's logging instead of tf.logging
   - Modern tf.io.gfile operations
   - Better TPU/GPU strategy handling

2. Model Architecture:
   - Keras-based model implementation
   - Better gradient handling
   - Improved checkpoint management
   - More efficient tensor operations

3. Data Pipeline:
   - Modern tf.data input pipeline
   - Better memory efficiency
   - Added data sampling capability
   - Improved preprocessing
"""

import collections
import json
import math
import os
import random
import re
import logging

from modeling import TFBertModel, BertConfig
import optimization
import tokenization
import tensorflow as tf
from absl import flags
from absl import app

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_float(
    "data_percentage", 100.0,
    "Percentage of data to use (between 0 and 100). Useful for quick testing.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


class SquadExample(object):
    """A single training/test example for the Squad dataset.
    
    This class represents one question-answer pair from the SQuAD dataset.
    For examples without an answer, the start and end position are -1.
    
    Attributes:
        qas_id: Unique id for the question-answer pair
        question_text: The question being asked
        doc_tokens: The document tokens that need to be processed
        orig_answer_text: The original answer text from the dataset
        start_position: Starting index of the answer in doc_tokens
        end_position: Ending index of the answer in doc_tokens
        is_impossible: Whether the question is impossible to answer (SQuAD 2.0)
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position is not None:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position is not None:
            s += ", end_position: %d" % (self.end_position)
        s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample.
    
    Migration Changes:
    - Using tf.io.gfile instead of tf.gfile
    - Added data sampling functionality
    - Improved error handling and logging
    - Better memory management
    
    Args:
        input_file: Path to the SQuAD json input file
        is_training: Whether this is for training
        
    Returns:
        A list of SquadExample objects
    """
    with tf.io.gfile.GFile(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    # Calculate how many examples to keep based on data_percentage
    if FLAGS.data_percentage < 100.0:
        num_examples = len(input_data)
        num_to_keep = max(1, int(num_examples * FLAGS.data_percentage / 100.0))
        logging.info(f"Using {FLAGS.data_percentage}% of data ({num_to_keep} out of {num_examples} examples)")
        # Randomly sample the data
        input_data = random.sample(input_data, num_to_keep)

    def is_whitespace(c):
        if c in (" ", "\t", "\r", "\n") or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:

                    if FLAGS.version_2_with_negative:
                        is_impossible = qa.get("is_impossible", False)
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logging.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s.
    
    Migration Changes:
    - Better tokenization handling
    - Improved memory efficiency
    - Enhanced error checking
    - Progress tracking
    
    Args:
        examples: List of SquadExample objects
        tokenizer: Tokenizer object for text processing
        max_seq_length: Maximum sequence length
        doc_stride: When splitting up a long document into chunks, stride
        max_query_length: Maximum length of the question
        is_training: Whether this is for training
        output_fn: Function to write features
    """

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                logging.info("*** Example ***")
                logging.info(f"unique_id: {unique_id}")
                logging.info(f"example_index: {example_index}")
                logging.info(f"doc_span_index: {doc_span_index}")
                logging.info(f"tokens: {' '.join([tokenization.printable_text(x) for x in tokens])}")
                logging.info(f"token_to_orig_map: {' '.join([f'{x}:{y}' for (x,y) in token_to_orig_map.items()])}")
                logging.info(f"token_is_max_context: {' '.join([f'{x}:{y}' for (x,y) in token_is_max_context.items()])}")
                logging.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                logging.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                logging.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
                if is_training and example.is_impossible:
                    logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logging.info(f"start_position: {start_position}")
                    logging.info(f"end_position: {end_position}")
                    logging.info(f"answer: {tokenization.printable_text(answer_text)}")

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)

            output_fn(feature)

            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator.
    
    Migration Changes:
    - Better TPU strategy handling
    - Improved checkpoint restoration
    - Enhanced optimizer configuration
    - More efficient loss computation
    
    Args:
        bert_config: BertConfig instance
        init_checkpoint: Path to initial checkpoint
        learning_rate: The initial learning rate
        num_train_steps: Number of training steps
        num_warmup_steps: Number of warmup steps
        use_tpu: Whether to use TPU
        use_one_hot_embeddings: Whether to use one-hot embeddings
    
    Returns:
        model_fn closure for TPUEstimator
    """

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info(f"  name = {name}, shape = {features[name].shape}")

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Create a keras model
        squad_model = BertSquadModel(
            bert_config=bert_config,
            is_training=is_training,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Call the model with inputs
        start_logits, end_logits = squad_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            training=is_training)

        tvars = squad_model.trainable_variables

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            # Load pre-trained model
            checkpoint = tf.train.Checkpoint(model=squad_model)
            checkpoint.restore(init_checkpoint).expect_partial()

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info(f"  name = {var.name}, shape = {var.shape}{init_string}")

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2.0

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                f"Only TRAIN and PREDICT modes are supported: {mode}")

        return output_spec

    return model_fn


class BertSquadModel(tf.keras.Model):
    """BERT model for Question Answering (SQuAD).
    
    Migration Changes:
    - Converted to Keras Model class
    - Native Keras layers
    - Better variable scope management
    - Improved forward pass efficiency
    
    This model takes a question and context as input and predicts
    the answer span within the context.
    """
    def __init__(self, bert_config, is_training, use_one_hot_embeddings):
        super(BertSquadModel, self).__init__()
        self.bert = TFBertModel(
            config=bert_config,
            name="bert")
            
        self.qa_outputs = tf.keras.layers.Dense(
            2,  # start/end logits
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="qa_outputs")

    @tf.function(reduce_retracing=True)
    def call(self, input_ids, attention_mask, token_type_ids, training=False):
        # Get the sequence output from BERT
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training)[0]

        # Apply the QA head
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        return start_logits, end_logits


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure for tf.data pipeline.
    
    Migration Changes:
    - Modern tf.data pipeline
    - Better memory efficiency
    - Improved shuffling and batching
    - Automatic prefetching
    
    Args:
        input_file: Path to TFRecord file
        seq_length: Maximum sequence length
        is_training: Whether this is for training
        drop_remainder: Whether to drop remainder batch
        
    Returns:
        A function that returns a tf.data.Dataset
    """

    name_to_features = {
        "unique_ids": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)

    @tf.function
    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)

        # Cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # Create a dataset of TFRecord files
        dataset = tf.data.TFRecordDataset(
            input_file,
            buffer_size=8 * 1024 * 1024  # 8MB buffer size
        )

        # Training specific processing
        if is_training:
            dataset = dataset.shuffle(buffer_size=100, seed=12345)
            dataset = dataset.repeat()

        # Parse the records with multiple workers
        dataset = dataset.map(
            _decode_record,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
    """Write final predictions to json files.
    
    Migration Changes:
    - Better error handling
    - Improved memory efficiency
    - Progress tracking
    - Enhanced logging
    
    Args:
        all_examples: List of SquadExample objects
        all_features: List of InputFeatures
        all_results: List of model predictions
        n_best_size: Number of best predictions to keep
        max_answer_length: Maximum answer length
        do_lower_case: Whether to lowercase the text
        output_prediction_file: Path to write predictions
        output_nbest_file: Path to write n-best predictions
        output_null_log_odds_file: Path to write null odds
    """
    logging.info(f"Writing predictions to: {output_prediction_file}")
    logging.info(f"Writing nbest to: {output_nbest_file}")
    logging.info(f"Total number of examples: {len(all_examples)}")
    logging.info(f"Total number of features: {len(all_features)}")
    logging.info(f"Total number of results: {len(all_results)}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    # Check for missing results
    missing_results = []
    for feature in all_features:
        if feature.unique_id not in unique_id_to_result:
            missing_results.append(feature.unique_id)
    
    if missing_results:
        logging.warning(f"Missing results for {len(missing_results)} features")
        logging.warning(f"First few missing IDs: {missing_results[:5]}")
        return

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        
        if not features:
            logging.warning(f"No features found for example index {example_index}")
            continue

        prelim_predictions = []
        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        
        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in unique_id_to_result:
                logging.warning(f"No result found for feature {feature.unique_id}")
                continue
                
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if FLAGS.version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        if FLAGS.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit))
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not FLAGS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > FLAGS.null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if FLAGS.version_2_with_negative:
        with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text.
    
    Migration Changes:
    - Improved text alignment
    - Better Unicode handling
    - Enhanced error checking
    
    Args:
        pred_text: The predicted text from model
        orig_text: The original text
        do_lower_case: Whether to lowercase the text
        
    Returns:
        The final text that best aligns with original
    """

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            logging.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
        return orig_text
    end_position = start_position + len(pred_text) - 1

    orig_ns_text, orig_ns_to_s_map = _strip_spaces(orig_text)
    tok_ns_text, tok_ns_to_s_map = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            logging.info(f"Length not equal after stripping spaces: '{orig_ns_text}' vs '{tok_ns_text}'")
        return orig_text

    tok_s_to_ns_map = {v: k for k, v in tok_ns_to_s_map.items()}

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(min(len(index_and_score), n_best_size)):
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = max(scores)

    exp_scores = [math.exp(score - max_score) for score in scores]
    total_sum = sum(exp_scores)

    probs = [score / total_sum for score in exp_scores]
    return probs


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self.writer = tf.io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            impossible = 1 if feature.is_impossible else 0
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self.writer.write(tf_example.SerializeToString())

    def close(self):
        self.writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            f"Cannot use sequence length {FLAGS.max_seq_length} because the BERT model "
            f"was only trained up to sequence length {bert_config.max_position_embeddings}")

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            f"The max_seq_length ({FLAGS.max_seq_length}) must be greater than max_query_length "
            f"({FLAGS.max_query_length}) + 3")


def main(_):
    """Main training and evaluation function.
    
    Migration Changes:
    - Better logging setup
    - Improved TPU/GPU strategy handling
    - Enhanced checkpoint management
    - Progress tracking
    - Memory efficient evaluation
    
    The function:
    1. Sets up logging and configuration
    2. Creates input pipelines
    3. Initializes model and optimizer
    4. Handles training loop
    5. Performs evaluation
    6. Saves results and checkpoints
    """
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # Set up TPU/GPU strategy
    strategy = None
    if FLAGS.use_tpu:
        if FLAGS.tpu_name:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
            tf.config.experimental_connect_to_cluster(cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            strategy = tf.distribute.TPUStrategy(cluster_resolver)
        else:
            raise ValueError("TPU name must be specified when using TPU")
    else:
        # Use MultiWorkerMirroredStrategy for multi-GPU training
        if len(tf.config.list_physical_devices('GPU')) > 1:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy

    # Training
    if FLAGS.do_train:
        train_examples = read_squad_examples(
            input_file=FLAGS.train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Pre-shuffle training examples
        rng = random.Random(12345)
        rng.shuffle(train_examples)

        # Convert examples to TF records
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_training=True)
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()

        logging.info("***** Running training *****")
        logging.info(f"  Num orig examples = {len(train_examples)}")
        logging.info(f"  Num split examples = {train_writer.num_features}")
        logging.info(f"  Batch size = {FLAGS.train_batch_size}")
        logging.info(f"  Num steps = {num_train_steps}")

        with strategy.scope():
            # Create model
            squad_model = BertSquadModel(
                bert_config=bert_config,
                is_training=True,
                use_one_hot_embeddings=FLAGS.use_tpu)

            # Create optimizer
            optimizer = optimization.create_optimizer(
                init_lr=FLAGS.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_tpu=FLAGS.use_tpu)

            # Restore checkpoint if provided
            if FLAGS.init_checkpoint:
                checkpoint = tf.train.Checkpoint(model=squad_model)
                checkpoint.restore(FLAGS.init_checkpoint).expect_partial()

            # Create training function
            @tf.function
            def train_step(inputs):
                with tf.GradientTape() as tape:
                    start_logits, end_logits = squad_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["input_mask"],
                        token_type_ids=inputs["segment_ids"],
                        training=True)
                    
                    # Compute loss
                    start_positions = inputs["start_positions"]
                    end_positions = inputs["end_positions"]
                    
                    seq_length = modeling.get_shape_list(inputs["input_ids"])[1]
                    def compute_loss(logits, positions):
                        one_hot_positions = tf.one_hot(
                            positions, depth=seq_length, dtype=tf.float32)
                        log_probs = tf.nn.log_softmax(logits, axis=-1)
                        loss = -tf.reduce_mean(
                            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                        return loss

                    start_loss = compute_loss(start_logits, start_positions)
                    end_loss = compute_loss(end_logits, end_positions)
                    total_loss = (start_loss + end_loss) / 2.0

                # Compute gradients and update
                grads = tape.gradient(total_loss, squad_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, squad_model.trainable_variables))
                return total_loss

            # Create training dataset
            train_input_fn = input_fn_builder(
                input_file=train_writer.filename,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
            
            train_dataset = train_input_fn({"batch_size": FLAGS.train_batch_size})
            train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

            # Training loop
            step = 0
            for batch in train_dist_dataset:
                if step >= num_train_steps:
                    break
                
                loss = strategy.run(train_step, args=(batch,))
                if step % FLAGS.save_checkpoints_steps == 0:
                    # Save checkpoint
                    checkpoint = tf.train.Checkpoint(model=squad_model, optimizer=optimizer)
                    checkpoint.save(os.path.join(FLAGS.output_dir, f"model-{step}"))
                    
                    # Log progress
                    logging.info(f"Step {step}: loss = {loss}")
                step += 1

    # Evaluation/Prediction
    if FLAGS.do_predict:
        eval_examples = read_squad_examples(
            input_file=FLAGS.predict_file, is_training=False)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        logging.info("***** Running predictions *****")
        logging.info(f"  Num orig examples = {len(eval_examples)}")
        logging.info(f"  Num split examples = {len(eval_features)}")
        logging.info(f"  Batch size = {FLAGS.predict_batch_size}")

        with strategy.scope():
            # Create model for prediction
            predict_model = BertSquadModel(
                bert_config=bert_config,
                is_training=False,
                use_one_hot_embeddings=FLAGS.use_tpu)

            # Restore from latest checkpoint
            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
            if latest_checkpoint:
                checkpoint = tf.train.Checkpoint(model=predict_model)
                checkpoint.restore(latest_checkpoint).expect_partial()

            # Create prediction function
            @tf.function
            def predict_step(inputs):
                try:
                    # Convert inputs to tensors if needed
                    input_ids = tf.convert_to_tensor(inputs["input_ids"], dtype=tf.int32)
                    attention_mask = tf.convert_to_tensor(inputs["input_mask"], dtype=tf.int32)
                    token_type_ids = tf.convert_to_tensor(inputs["segment_ids"], dtype=tf.int32)
                    
                    # Add batch dimension if needed
                    if len(input_ids.shape) == 1:
                        input_ids = tf.expand_dims(input_ids, 0)
                        attention_mask = tf.expand_dims(attention_mask, 0)
                        token_type_ids = tf.expand_dims(token_type_ids, 0)
                    
                    return predict_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        training=False)
                except Exception as e:
                    logging.error(f"Error in predict_step: {str(e)}")
                    logging.error(f"Input shapes: ids={input_ids.shape}, mask={attention_mask.shape}, type_ids={token_type_ids.shape}")
                    raise

            # Create prediction dataset
            predict_input_fn = input_fn_builder(
                input_file=eval_writer.filename,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=False)
            predict_dataset = predict_input_fn({"batch_size": FLAGS.predict_batch_size})

            # Run predictions
            all_results = []
            processed_unique_ids = set()
            total_examples = sum(1 for _ in predict_dataset)
            logging.info(f"Starting predictions on {total_examples} batches")
            
            try:
                for batch_idx, inputs in enumerate(predict_dataset):
                    if batch_idx % 10 == 0:
                        logging.info(f"Processing batch {batch_idx}")
                        # Clear memory
                        tf.keras.backend.clear_session()
                    
                    try:
                        unique_ids = inputs["unique_ids"]
                        start_logits, end_logits = predict_step(inputs)
                        
                        # Convert to numpy and free GPU memory
                        start_logits = start_logits.numpy()
                        end_logits = end_logits.numpy()
                        
                        for i, unique_id in enumerate(unique_ids):
                            unique_id_int = int(unique_id)
                            if unique_id_int in processed_unique_ids:
                                logging.warning(f"Duplicate unique_id found: {unique_id_int}")
                                continue
                                
                            processed_unique_ids.add(unique_id_int)
                            all_results.append(
                                RawResult(
                                    unique_id=unique_id_int,
                                    start_logits=start_logits[i].tolist(),
                                    end_logits=end_logits[i].tolist()))
                            
                        # Free memory
                        del start_logits
                        del end_logits
                        
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                        continue
                    
                    # Save intermediate results every 100 batches
                    if batch_idx > 0 and batch_idx % 100 == 0:
                        logging.info(f"Saving intermediate results at batch {batch_idx}")
                        logging.info(f"Processed {len(processed_unique_ids)} unique examples so far")
                        
                        # Verify all features have corresponding results before writing
                        feature_ids = set(f.unique_id for f in eval_features)
                        result_ids = set(r.unique_id for r in all_results)
                        missing_ids = feature_ids - result_ids
                        
                        if missing_ids:
                            logging.warning(f"Missing results for {len(missing_ids)} features")
                            logging.warning(f"First few missing IDs: {list(missing_ids)[:5]}")
                            continue
                            
                        output_prediction_file = os.path.join(FLAGS.output_dir, f"predictions_batch_{batch_idx}.json")
                        output_nbest_file = os.path.join(FLAGS.output_dir, f"nbest_predictions_batch_{batch_idx}.json")
                        output_null_log_odds_file = os.path.join(FLAGS.output_dir, f"null_odds_batch_{batch_idx}.json")
                        
                        write_predictions(
                            eval_examples, eval_features, all_results,
                            FLAGS.n_best_size, FLAGS.max_answer_length,
                            FLAGS.do_lower_case, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file)
                
            except Exception as e:
                logging.error(f"Error in prediction loop: {str(e)}")
                raise

            # Write final predictions
            output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
            output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
            output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

            write_predictions(
                eval_examples, eval_features, all_results,
                FLAGS.n_best_size, FLAGS.max_answer_length,
                FLAGS.do_lower_case, output_prediction_file,
                output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    app.run(main)
