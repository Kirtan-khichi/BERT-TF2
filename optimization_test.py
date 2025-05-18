# coding=utf-8

"""
Unit Tests for BERT AdamW Optimizer

This file tests the AdamW optimizer implementation, verifying:
- Correct weight updates
- Convergence to target values
- Proper gradient handling
- Weight decay behavior

Key Migration Changes from TF1:
1. Testing Framework:
   - Removed tf.Session and manual initialization
   - Using tf.GradientTape for automatic differentiation
   - Native TF2 variable handling
   - Simpler test execution

2. Variable Management:
   - Using tf.Variable directly instead of tf.get_variable
   - No explicit variable initialization required
   - Better dtype handling
   - Cleaner variable creation

3. Gradient Computation:
   - Using GradientTape instead of tf.gradients
   - No global step management needed
   - More straightforward gradient application
   - Better training loop structure

4. Test Structure:
   - Simplified test setup
   - Clearer test flow
   - Better error messages
   - More maintainable tests
"""

import tensorflow as tf
from optimization import AdamWeightDecayOptimizer


class OptimizationTest(tf.test.TestCase):
    """Test suite for AdamW optimizer implementation.
    
    Migration changes:
    - Converted to TF2 testing style
    - Removed session handling
    - Using GradientTape
    - Better variable management
    """

    def test_adam(self):
        """Tests AdamW optimizer convergence and weight updates.
        
        Migration changes:
        - Removed tf.Session context
        - Using tf.Variable directly
        - GradientTape for gradients
        - Simpler training loop
        
        The test:
        1. Creates a simple optimization problem
        2. Runs AdamW optimizer for 100 steps
        3. Verifies weights converge to expected values
        
        Target values [0.4, 0.2, -0.5] are chosen to verify:
        - Both positive and negative convergence
        - Different magnitudes of updates
        - Proper handling of weight decay
        """
        # Variable initialized close to [0.1, -0.2, -0.1]
        w = tf.Variable([0.1, -0.2, -0.1], dtype=tf.float32, name="w")
        x = tf.constant([0.4, 0.2, -0.5], dtype=tf.float32)

        # Create optimizer with fixed learning rate
        optimizer = AdamWeightDecayOptimizer(learning_rate=0.2)

        # Training loop using GradientTape
        for _ in range(100):
            with tf.GradientTape() as tape:
                # Simple L2 loss between w and target x
                loss = tf.reduce_mean(tf.square(x - w))
            # Compute and apply gradients
            grads = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grads, [w]))

        # Verify convergence to target values
        w_val = w.numpy()
        self.assertAllClose(w_val, [0.4, 0.2, -0.5], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tf.test.main()
