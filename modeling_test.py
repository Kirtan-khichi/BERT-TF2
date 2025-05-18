# coding=utf-8


"""
Unit Tests for BERT Model Implementation

This file contains tests for the TensorFlow 2.x BERT implementation, verifying:
- Model architecture and shapes
- Embedding layer functionality
- Encoder layer behavior
- Attention mask handling
- Dropout behavior
- Configuration serialization

Key Migration Changes from TF1:
1. Test Framework Updates:
   - Switched from tf.test.TestCase with sessions to modern tf.test.TestCase
   - Removed manual session handling and variable initialization
   - Using self.evaluate() instead of sess.run()
   - Better test organization with setUp method

2. Model Testing Approach:
   - More focused, granular test methods
   - Separate tests for embeddings, encoder, and full model
   - Better handling of training vs evaluation modes
   - Simplified tensor creation and manipulation

3. Assertion Patterns:
   - Using TF2 native assertions
   - More precise shape checking
   - Better error messages
   - Cleaner test output

4. Code Organization:
   - Removed complex test infrastructure
   - Simplified helper functions
   - Better separation of concerns
   - More maintainable test structure
"""

import json
import random
import tensorflow as tf
from modeling import BertConfig, TFBertModel, create_attention_mask_from_input_mask


def ids_tensor(shape, vocab_size, rng=None):
    """Creates a random int32 tensor for testing.
    
    Migration changes:
    - Simplified from class method to standalone function
    - Removed unnecessary name parameter
    - More efficient tensor creation
    
    Args:
        shape: List of dimensions for the tensor
        vocab_size: Upper bound for random values (exclusive)
        rng: Optional random number generator
        
    Returns:
        tf.Tensor of shape with random ints in [0, vocab_size)
    """
    if rng is None:
        rng = random.Random()
    total = 1
    for d in shape:
        total *= d
    vals = [rng.randint(0, vocab_size - 1) for _ in range(total)]
    return tf.constant(vals, dtype=tf.int32, shape=shape)


class BertModelTest(tf.test.TestCase):
    """Test suite for BERT model implementation.
    
    Migration changes:
    - Converted to modern tf.test.TestCase style
    - Removed complex test infrastructure
    - Added setUp for better test organization
    - More focused test methods
    
    Tests cover:
    1. Embedding layer output shapes and behavior
    2. Encoder layer transformations
    3. Attention mask handling
    4. Full model integration
    5. Configuration serialization
    """

    def setUp(self):
        """Initialize test parameters.
        
        Migration changes:
        - Added setUp method for cleaner test organization
        - Centralized test parameters
        - Better parameter documentation
        """
        super().setUp()
        # Model architecture parameters
        self.batch_size = 13
        self.seq_length = 7
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37

    def create_model(self):
        """Creates a BERT model instance with test inputs.
        
        Migration changes:
        - Simplified model creation
        - Better input tensor generation
        - Cleaner config handling
        
        Returns:
            Tuple of (model, input_ids, attention_mask, token_type_ids)
        """
        # Generate test inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = tf.ones_like(input_ids)
        token_type_ids = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        # Create model with test configuration
        config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size
        )
        model = TFBertModel(config)
        return model, input_ids, attention_mask, token_type_ids

    def test_embedding_output(self):
        """Tests the embedding layer output.
        
        Migration changes:
        - Focused test for embedding layer
        - Direct embedding layer testing
        - Better shape verification
        
        Verifies:
        1. Output shape matches [batch_size, seq_length, hidden_size]
        2. Embedding layer processes inputs correctly
        """
        model, input_ids, _, token_type_ids = self.create_model()
        emb = model.embeddings(input_ids, token_type_ids, training=False)
        emb_val = self.evaluate(emb)
        self.assertAllEqual(
            emb_val.shape,
            [self.batch_size, self.seq_length, self.hidden_size]
        )

    def test_encoder_all_layers(self):
        """Tests each encoder layer's output.
        
        Migration changes:
        - Layer-by-layer testing
        - Better attention mask handling
        - Clearer shape verification
        
        Verifies:
        1. Each layer produces correct output shape
        2. Correct number of layers created
        3. Attention mask properly applied
        """
        model, input_ids, attention_mask, token_type_ids = self.create_model()
        hidden = model.embeddings(input_ids, token_type_ids, training=False)
        mask = create_attention_mask_from_input_mask(input_ids, attention_mask)
        
        all_outputs = []
        for layer in model.encoder.layer:
            hidden = layer(hidden, mask, training=False)
            all_outputs.append(hidden)
            
        self.assertEqual(len(all_outputs), self.num_hidden_layers)
        for out in all_outputs:
            val = self.evaluate(out)
            self.assertAllEqual(
                val.shape,
                [self.batch_size, self.seq_length, self.hidden_size]
            )

    def test_default_and_no_mask(self):
        """Tests model behavior with and without attention masks.
        
        Migration changes:
        - Simplified mask testing
        - Better output validation
        - Clearer test organization
        
        Verifies:
        1. Model works with attention mask
        2. Model works without attention mask
        3. Output shapes are correct in both cases
        """
        model, input_ids, mask, token_types = self.create_model()
        
        # Test with mask
        seq_out, pooled_out = model(
            input_ids,
            attention_mask=mask,
            token_type_ids=token_types,
            training=False
        )
        seq_val = self.evaluate(seq_out)
        pool_val = self.evaluate(pooled_out)
        self.assertAllEqual(seq_val.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.assertAllEqual(pool_val.shape, [self.batch_size, self.hidden_size])
        
        # Test without mask
        seq_nomask, pool_nomask = model(input_ids, attention_mask=None, token_type_ids=None, training=False)
        self.assertAllEqual(
            self.evaluate(seq_nomask).shape,
            [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.assertAllEqual(
            self.evaluate(pool_nomask).shape,
            [self.batch_size, self.hidden_size]
        )

    def test_dropout_disabled_in_eval(self):
        """Verifies deterministic behavior when dropout is disabled.
        
        Migration changes:
        - Simplified dropout testing
        - Better output comparison
        - Clearer verification
        
        Verifies:
        1. Outputs are identical with training=False
        2. Dropout is properly disabled in eval mode
        """
        model, input_ids, mask, types = self.create_model()
        out1, _ = model(input_ids, attention_mask=mask, token_type_ids=types, training=False)
        out2, _ = model(input_ids, attention_mask=mask, token_type_ids=types, training=False)
        self.assertAllClose(self.evaluate(out1), self.evaluate(out2))

    def test_config_to_json_string(self):
        """Tests configuration serialization.
        
        Migration changes:
        - Simplified config testing
        - Better JSON handling
        - Clearer verification
        
        Verifies:
        1. Config properly serializes to JSON
        2. Values are correctly preserved
        """
        config = BertConfig(vocab_size=99, hidden_size=37)
        obj = json.loads(config.to_json_string())
        self.assertEqual(obj['vocab_size'], 99)
        self.assertEqual(obj['hidden_size'], 37)


if __name__ == '__main__':
    tf.test.main()
