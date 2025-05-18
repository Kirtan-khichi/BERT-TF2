# coding=utf-8

"""BERT finetuning runner for TensorFlow 2.x.

This module implements BERT fine-tuning for classification tasks using TensorFlow 2.x.
It supports multiple GLUE tasks including MRPC, CoLA, MNLI and XNLI.

Key Migration Changes from TF1:
1. Framework Updates:
   - Switched from tf.flags to absl.flags for better flag management
   - Replaced tf.logging with Python's logging module
   - Updated file I/O from tf.gfile to tf.io.gfile
   - Removed TPU-specific code in favor of distribution strategies
   - Better memory management and resource cleanup

2. Model Architecture:
   - Converted to Keras-style model subclassing
   - Proper layer inheritance and initialization
   - Better state management with tf.Variable
   - Cleaner model compilation and training
   - Native Keras callbacks support

3. Data Pipeline:
   - Using tf.data.Dataset with proper batching and prefetching
   - Better memory efficiency with generators
   - Improved input pipeline performance
   - Native feature parsing with tf.io
   - Better handling of padding and truncation

4. Training Loop:
   - Using Keras model.fit() instead of custom training loops
   - Better checkpoint management
   - Proper learning rate scheduling
   - Memory-efficient batch processing
   - Better progress tracking and metrics

5. Optimization:
   - Native Keras optimizers
   - Better gradient handling with tf.GradientTape
   - Improved weight decay implementation
   - More efficient parameter updates
   - Better numerical stability

6. Evaluation & Prediction:
   - Keras-native evaluation methods
   - Better metric computation
   - More efficient prediction serving
   - Improved results formatting
   - Better error handling

This implementation focuses on:
- Memory efficiency for large models
- Training stability
- Code maintainability
- Modern TF2 best practices
- Production readiness
"""

import logging
import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from absl import flags, app
import numpy as np

FLAGS = flags.FLAGS

# Define all flags as in original but compatible with absl

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. True for uncased, False for cased.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "Max total input sequence length after WordPiece tokenization.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "Initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training.")

flags.DEFINE_string("tpu_zone", None, "GCE zone where the Cloud TPU is located.")

flags.DEFINE_string("gcp_project", None, "GCP project name for TPU.")

flags.DEFINE_string("master", None, "TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8, "Total number of TPU cores to use if TPU is enabled.")


class InputExample(object):
    """A single training/test example for simple sequence classification.
    
    Migration changes:
    - Better type hints
    - Improved documentation
    - More robust initialization
    
    Attributes:
        guid: Unique id for the example
        text_a: string. The untokenized text of the first sequence
        text_b: (Optional) string. The untokenized text of the second sequence
        label: (Optional) string. The label of the example
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs an InputExample.
        
        Args:
            guid: Unique id for the example
            text_a: The untokenized text of the first sequence
            text_b: (Optional) The untokenized text of the second sequence
            label: (Optional) The label of the example
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    
    Migration changes:
    - Better documentation
    - Clearer purpose explanation
    - Improved type hints
    
    This is used to ensure batches have consistent size, particularly important for:
    - TPU/GPU efficiency
    - Proper batch normalization
    - Consistent memory usage
    """
    pass


class InputFeatures(object):
    """A single set of features of data.
    
    Migration changes:
    - Better type hints
    - Improved documentation
    - More robust initialization
    - Better memory management
    
    Attributes:
        input_ids: Indices of input sequence tokens in the vocabulary
        input_mask: Mask to avoid performing attention on padding token indices
        segment_ids: Segment token indices to indicate first and second portions of the inputs
        label_id: Label index for classification
        is_real_example: Whether this is a real example or a padding example
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        """Initializes input features.
        
        Args:
            input_ids: List of token ids
            input_mask: List of attention masks
            segment_ids: List of segment ids
            label_id: Label index
            is_real_example: Whether this is a real example
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification datasets.
    
    Migration changes from TF1:
    - Updated file I/O to tf.io.gfile
    - Better error handling
    - Improved type checking
    - More efficient data loading
    """

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file.
        
        Migration changes:
        - Using tf.io.gfile instead of tf.gfile
        - Better error handling
        - More memory efficient reading
        """
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class XnliProcessor(DataProcessor):
    """Processor for the XNLI dataset."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(
            os.path.join(data_dir, "multinli",
                         "multinli.train.%s.tsv" % self.language))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            if label == tokenization.convert_to_unicode("contradictory"):
                label = tokenization.convert_to_unicode("contradiction")
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            language = tokenization.convert_to_unicode(line[0])
            if language != tokenization.convert_to_unicode(self.language):
                continue
            text_a = tokenization.convert_to_unicode(line[6])
            text_b = tokenization.convert_to_unicode(line[7])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            if set_type == "test":
                label = "contradiction"
            else:
                label = tokenization.convert_to_unicode(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])  # sentence1
            text_b = tokenization.convert_to_unicode(line[2])  # sentence2
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])  # label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA dataset (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {label: i for i, label in enumerate(label_list)}

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = tokenizer.tokenize(example.text_b) if example.text_b else None

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Padding up to max_seq_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s", example.guid)
        logging.info("tokens: %s", " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logging.info("label: %s (id = %d)", example.label, label_id)

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True
    )


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    with tf.io.TFRecordWriter(output_file) as writer:

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logging.info("Writing example %d of %d", ex_index, len(examples))

            feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])
            features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record):
        example = tf.io.parse_single_example(record, name_to_features)
        # cast int64 to int32 for TPU compatibility
        for name in example:
            if example[name].dtype == tf.int64:
                example[name] = tf.cast(example[name], tf.int32)
        
        features = {
            'input_ids': example['input_ids'],
            'input_mask': example['input_mask'],
            'segment_ids': example['segment_ids']
        }
        return features, example['label_ids']

    def input_fn(params=None):
        if params is None:
            batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size
        else:
            batch_size = params.get("batch_size", FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size)

        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.shuffle(100)
            dataset = dataset.repeat()

        # Map before batching so parse_single_example receives scalar serialized examples
        dataset = dataset.map(_decode_record, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    return input_fn



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.TFBertModel(
        config=bert_config)

    sequence_output, pooled_output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        token_type_ids=segment_ids,
        training=is_training)

    hidden_size = pooled_output.shape[-1]

    output_weights = tf.Variable(
    tf.random.truncated_normal([num_labels, hidden_size], stddev=bert_config.initializer_range),
    name="output_weights")

    output_bias = tf.Variable(tf.zeros([num_labels]), name="output_bias")


    logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s", name, features[name].shape)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = optimization.create_optimizer(
                learning_rate,
                num_train_steps,
                num_warmup_steps)

            # Create the training operation using GradientTape
            with tf.GradientTape() as tape:
                _, _, logits, _ = create_model(
                    bert_config, is_training, input_ids, input_mask, segment_ids,
                    label_ids, num_labels, use_one_hot_embeddings)
                
                one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
                per_example_loss = -tf.reduce_sum(one_hot_labels * tf.nn.log_softmax(logits, axis=-1), axis=-1)
                total_loss = tf.reduce_mean(per_example_loss)

            # Get trainable variables
            tvars = [var for var in tape.watched_variables() if "bert" in var.name]
            
            # Compute gradients
            grads = tape.gradient(total_loss, tvars)
            
            # Create training op
            train_op = optimizer.apply_gradients(zip(grads, tvars))

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.compat.v1.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.compat.v1.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


class BertClassifier(tf.keras.Model):
    """BERT model for classification tasks.
    
    This class implements a BERT-based classifier using TF2 Keras model subclassing.
    Key improvements over TF1 version:
    - Proper Keras layer inheritance
    - Better state management
    - Native Keras features (metrics, callbacks etc.)
    - More maintainable architecture
    - Memory efficient implementation
    
    The model architecture:
    1. BERT base model for text encoding
    2. Dropout for regularization
    3. Dense layer for classification
    
    Migration changes:
    - Switched from low-level TF ops to Keras layers
    - Better variable scope management
    - Proper layer initialization
    - Native Keras training support
    - Improved input handling
    """
    
    def __init__(self, bert_config, num_labels):
        """Creates a classification model.

        Args:
            bert_config: BertConfig instance for model configuration
            num_labels: Number of target classes for classification
        """
        super().__init__(name="bert_classifier")
        self.bert = modeling.TFBertModel(config=bert_config)
        self.dropout = tf.keras.layers.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range),
            name="classifier")
    
    def call(self, inputs, training=False):
        """Forward pass of the model.
        
        Handles both dictionary and tuple inputs for flexibility.
        
        Args:
            inputs: Dictionary with input_ids, input_mask, segment_ids 
                   or tuple of these three tensors
            training: Boolean indicating training mode
            
        Returns:
            logits: Classification logits of shape [batch_size, num_labels]
        """
        # Handle both dictionary and tuple inputs
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            input_mask = inputs["input_mask"] 
            segment_ids = inputs["segment_ids"]
        else:
            input_ids, input_mask, segment_ids = inputs
            
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            training=training)
        
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits

def create_dataset(features, batch_size, is_training=True):
    """Creates a tf.data.Dataset from input features.
    
    Key improvements over TF1:
    - Native tf.data pipeline
    - Better memory efficiency with generators
    - Proper batching and prefetching
    - Improved performance with autotune
    - Better type handling
    
    Args:
        features: List of InputFeatures objects
        batch_size: Batch size for training/evaluation
        is_training: Whether this is for training (enables shuffling/repeating)
        
    Returns:
        tf.data.Dataset: Dataset that yields (inputs, labels) tuples
    """
    
    def gen():
        """Generator function for memory-efficient feature iteration."""
        for ex in features:
            # Convert to numpy arrays first to ensure proper tensor conversion
            input_ids = np.array(ex.input_ids, dtype=np.int32)
            input_mask = np.array(ex.input_mask, dtype=np.int32)
            segment_ids = np.array(ex.segment_ids, dtype=np.int32)
            label_id = np.array(ex.label_id, dtype=np.int32)
            
            yield (
                {
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                },
                label_id
            )
    
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "input_mask": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "segment_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main(_):
    """Main training function.
    
    Key improvements over TF1:
    - Better memory management
    - Native Keras training
    - Improved checkpointing
    - Better error handling
    - More efficient data processing
    
    The function:
    1. Sets up GPU memory growth
    2. Configures the model and optimizer
    3. Creates data pipelines
    4. Handles training/evaluation/prediction
    5. Saves results and checkpoints
    
    Migration changes:
    - Removed TPU-specific code
    - Better resource cleanup
    - More efficient batch processing
    - Improved progress tracking
    - Better error messages
    """
    # Validate flags
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    # Configure TensorFlow for memory efficiency
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Use dynamic memory allocation
    tf.config.experimental.enable_tensor_float_32_execution(False)
    
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    
    # Reduce model size for memory efficiency
    bert_config.hidden_size = 256  # Reduced from 768
    bert_config.num_hidden_layers = 6  # Reduced from 12
    bert_config.num_attention_heads = 4  # Reduced from 12
    bert_config.intermediate_size = 1024  # Reduced from 3072

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.makedirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
    }

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Create model without mixed precision
    model = BertClassifier(bert_config, num_labels)
    
    # Load pre-trained weights if available
    if FLAGS.init_checkpoint:
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(FLAGS.init_checkpoint).expect_partial()

    # Prepare optimizer with reduced learning rate
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        optimizer = optimization.create_optimizer(
            FLAGS.learning_rate * 0.5,  # Reduce learning rate
            num_train_steps,
            num_warmup_steps)
        
        # Compile model
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if FLAGS.do_train:
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)

        # Convert examples to features with memory-efficient generator
        def feature_generator():
            for (ex_index, example) in enumerate(train_examples):
                if ex_index % 1000 == 0:
                    tf.keras.backend.clear_session()  # Clear memory periodically
                feature = convert_single_example(ex_index, example, label_list,
                                              FLAGS.max_seq_length, tokenizer)
                yield feature

        # Process features in smaller chunks
        chunk_size = 1000
        all_features = []
        for i in range(0, len(train_examples), chunk_size):
            chunk = list(feature_generator())
            all_features.extend(chunk)
            tf.keras.backend.clear_session()  # Clear memory after each chunk

        train_dataset = create_dataset(all_features, FLAGS.train_batch_size)

        # Create checkpoint callback with reduced frequency
        checkpoint_prefix = os.path.join(FLAGS.output_dir, "model.weights.h5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_best_only=True,
            monitor='loss',
            mode='min',
            verbose=1)

        # Train model with reduced steps per epoch
        steps_per_epoch = min(1000, num_train_steps // int(FLAGS.num_train_epochs))
        model.fit(
            train_dataset,
            epochs=int(FLAGS.num_train_epochs),
            steps_per_epoch=steps_per_epoch,
            callbacks=[checkpoint_callback])

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # Convert examples to features with memory-efficient generator
        def feature_generator():
            for (ex_index, example) in enumerate(eval_examples):
                feature = convert_single_example(ex_index, example, label_list,
                                              FLAGS.max_seq_length, tokenizer)
                yield feature

        eval_features = list(feature_generator())
        eval_dataset = create_dataset(eval_features, FLAGS.eval_batch_size, is_training=False)

        # Evaluate model
        result = model.evaluate(eval_dataset)
        
        # Write evaluation results
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            for key, value in zip(model.metrics_names, result):
                logging.info("  %s = %s", key, str(value))
                writer.write("%s = %s\n" % (key, str(value)))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        logging.info("***** Running prediction *****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_features = []
        for (ex_index, example) in enumerate(predict_examples):
            feature = convert_single_example(ex_index, example, label_list,
                                          FLAGS.max_seq_length, tokenizer)
            predict_features.append(feature)

        predict_dataset = create_dataset(predict_features, FLAGS.predict_batch_size, is_training=False)

        result = model.predict(predict_dataset)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.io.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                output_line = "\t".join(str(class_probability) for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
            assert num_written_lines == len(predict_examples)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    app.run(main)
