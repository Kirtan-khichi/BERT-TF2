# coding=utf-8

"""BERT finetuning runner with TF-Hub for TensorFlow 2.x."""

import os
import logging
import optimization
import run_classifier
import tokenization

# SSL Configuration - Must be before importing tensorflow and tensorflow_hub
import ssl
import urllib.request
import tempfile
import certifi

# Create an SSL context that doesn't verify certificates
ssl_context = ssl._create_unverified_context()

# Store the original urlopen
original_urlopen = urllib.request.urlopen

# Configure urllib to use the unverified context
def urlopen_with_retry(url, *args, **kwargs):
    if 'context' in kwargs:
        del kwargs['context']
    return original_urlopen(url, *args, context=ssl_context, **kwargs)

# Replace urllib's urlopen with our custom function
urllib.request.urlopen = urlopen_with_retry

# Now import tensorflow and tensorflow_hub
import tensorflow as tf
import tensorflow_hub as hub
from absl import flags, app

# Modify TensorFlow Hub's resolver to use our SSL context
import tensorflow_hub.resolver as resolver
resolver._call_urlopen = urlopen_with_retry

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_hub_module_handle", None,
    "Handle for the BERT TF-Hub module.")

def download_vocab_from_hub(bert_hub_module_handle):
    """Downloads the vocab file from TF Hub module."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bert_layer = hub.KerasLayer(bert_hub_module_handle, trainable=True)
        
        # Get the resolved object path
        resolved_object = bert_layer.resolved_object
        
        # Try different ways to get the vocab file
        try:
            # First try the new way (TF 2.x models)
            vocab_file = resolved_object.vocab_file.asset_path.numpy()
            if isinstance(vocab_file, bytes):
                vocab_file = vocab_file.decode('utf-8')
        except:
            try:
                # Try the old way (TF 1.x models)
                assets = tf.io.gfile.listdir(os.path.join(resolved_object._path, "assets"))
                vocab_file = None
                for asset in assets:
                    if 'vocab.txt' in asset:
                        vocab_file = os.path.join(resolved_object._path, "assets", asset)
                        break
                if vocab_file is None:
                    raise ValueError("Could not find vocab.txt in model assets")
            except:
                # Try one more way
                try:
                    vocab_file = os.path.join(resolved_object._path, "assets", "vocab.txt")
                    if not tf.io.gfile.exists(vocab_file):
                        raise ValueError("Vocab file does not exist at expected path")
                except:
                    raise ValueError("Could not find vocab file through any known method")
        
        # Copy vocab file to a temporary location
        local_vocab_file = os.path.join(tmp_dir, "vocab.txt")
        tf.io.gfile.copy(vocab_file, local_vocab_file, overwrite=True)
        
        # Read the contents of the vocab file
        with tf.io.gfile.GFile(local_vocab_file, 'rb') as f:
            vocab_content = f.read()
            
        # Create a new temporary file that will persist
        final_vocab_file = os.path.join(tempfile.gettempdir(), 'vocab.txt')
        with open(final_vocab_file, 'wb') as f:
            f.write(vocab_content)
            
        return final_vocab_file

class BertClassifierModel(tf.keras.Model):
    """BERT classifier model."""
    
    def __init__(self, num_labels, bert_hub_module_handle, trainable=True):
        super().__init__(name="bert_classifier")
        self.num_labels = num_labels
        
        # Load BERT layer from TF-Hub
        self.bert = hub.KerasLayer(bert_hub_module_handle, trainable=trainable)
        
        # Add classification head
        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        self.classifier = tf.keras.layers.Dense(
            num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="classifier")

    def call(self, inputs, training=False):
        # Unpack inputs
        input_ids = inputs["input_ids"]
        input_mask = inputs["input_mask"]
        segment_ids = inputs["segment_ids"]
        
        # Get BERT outputs
        bert_outputs = self.bert({
            "input_word_ids": input_ids,
            "input_mask": input_mask,
            "input_type_ids": segment_ids
        }, training=training)
        
        # Use pooled output for classification
        pooled_output = bert_outputs["pooled_output"]
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        
        return logits

def create_model(num_labels, bert_hub_module_handle, learning_rate):
    """Creates and compiles the BERT classifier model."""
    
    # Create model
    model = BertClassifierModel(num_labels, bert_hub_module_handle)
    
    # Compile model
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_tokenizer_from_hub_module(bert_hub_module_handle):
    """Get the vocab file and casing info from the Hub module."""
    vocab_file = download_vocab_from_hub(bert_hub_module_handle)
    # BERT uncased model always uses do_lower_case=True
    do_lower_case = True
    
    return tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case
    )

def main(_):
    # Configure logging
    logger.setLevel(logging.INFO)

    processors = {
        "cola": run_classifier.ColaProcessor,
        "mnli": run_classifier.MnliProcessor,
        "mrpc": run_classifier.MrpcProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    tf.io.gfile.makedirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Use local model path instead of URL
    tokenizer = create_tokenizer_from_hub_module(FLAGS.bert_hub_module_handle)

    # Create the model
    model = create_model(
        num_labels=num_labels,
        bert_hub_module_handle=FLAGS.bert_hub_module_handle,
        learning_rate=FLAGS.learning_rate
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Update optimizer with correct number of steps
        model.optimizer = optimization.create_optimizer(
            init_lr=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps
        )

        # Convert examples to features and create TF records
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        run_classifier.file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

        # Create training dataset
        train_dataset = run_classifier.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)({"batch_size": FLAGS.train_batch_size})

        # Create checkpoint callback with correct file format
        checkpoint_prefix = os.path.join(FLAGS.output_dir, "model.weights.h5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_best_only=True,
            monitor='loss')

        # Train the model
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_examples)}")
        logger.info(f"  Batch size = {FLAGS.train_batch_size}")
        logger.info(f"  Num steps = {num_train_steps}")

        model.fit(
            train_dataset,
            epochs=int(FLAGS.num_train_epochs),
            steps_per_epoch=num_train_steps // int(FLAGS.num_train_epochs),
            callbacks=[checkpoint_callback]
        )

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        run_classifier.file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        # Create eval dataset
        eval_dataset = run_classifier.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)({"batch_size": FLAGS.eval_batch_size})

        # Evaluate the model
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(eval_examples)}")
        logger.info(f"  Batch size = {FLAGS.eval_batch_size}")

        results = model.evaluate(eval_dataset)

        # Write evaluation results
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for name, value in zip(model.metrics_names, results):
                logger.info(f"  {name} = {value}")
                writer.write(f"{name} = {value}\n")

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        run_classifier.file_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file)

        # Create predict dataset
        predict_dataset = run_classifier.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)({"batch_size": FLAGS.predict_batch_size})

        # Run prediction
        logger.info("***** Running prediction *****")
        logger.info(f"  Num examples = {len(predict_examples)}")
        logger.info(f"  Batch size = {FLAGS.predict_batch_size}")

        predictions = model.predict(predict_dataset)

        # Write predictions
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.io.gfile.GFile(output_predict_file, "w") as writer:
            logger.info("***** Predict results *****")
            for prediction in predictions:
                probabilities = tf.nn.softmax(prediction)
                output_line = "\t".join(str(p) for p in probabilities.numpy()) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("bert_hub_module_handle")
    flags.mark_flag_as_required("output_dir")
    app.run(main)
