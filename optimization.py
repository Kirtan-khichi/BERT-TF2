# coding=utf-8

"""
BERT Optimizer Implementation for TensorFlow 2.x

This module implements the AdamW optimizer (Adam with correct weight decay)
and learning rate scheduling used in BERT training.

Key Migration Changes from TF1:
1. Optimizer Implementation:
   - Inherits from tf.keras.optimizers.Adam instead of tf.train.Optimizer
   - Uses Keras-native optimizer methods and states
   - Better handling of weight decay through Keras mechanisms
   - Proper serialization support with get_config

2. Learning Rate Scheduling:
   - Switched to tf.keras.optimizers.schedules
   - Separated warmup into its own schedule class
   - Better handling of learning rate updates
   - More efficient step counting

3. Training Loop Integration:
   - Removed TPU-specific code (handled by distribution strategy)
   - No manual global step management
   - Better gradient handling
   - Native Keras training loop support

4. Code Organization:
   - More modular design
   - Better separation of concerns
   - Cleaner class hierarchy
   - More maintainable structure

Original paper: https://arxiv.org/abs/1810.04805
"""

import re
import tensorflow as tf


class AdamWeightDecayOptimizer(tf.keras.optimizers.Adam):
    """AdamW optimizer (Adam with correct L2 weight decay).
    
    Migration changes from TF1:
    - Inherits from tf.keras.optimizers.Adam instead of tf.train.Optimizer
    - Uses Keras optimizer infrastructure
    - Better state management
    - Proper serialization
    
    This implements the AdamW algorithm (https://arxiv.org/abs/1711.05101),
    which performs L2 weight decay differently from the regularization in 
    standard Adam:
    - Standard Adam: Weight decay is part of gradient computation
    - AdamW: Weight decay is applied separately from gradient update
    
    The difference matters because Adam's momentum and variance 
    calculations prevent the weight decay from directly affecting the weights.
    """

    def __init__(self,
                learning_rate=0.001,
                weight_decay_rate=0.0,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=None,
                name="AdamWeightDecayOptimizer"):
        """Initializes the optimizer.
        
        Migration changes:
        - Uses Keras-style initialization
        - Better parameter handling
        - Proper name scoping
        
        Args:
            learning_rate: A float, a schedule, or a callable.
            weight_decay_rate: Weight decay rate for L2 regularization.
            beta_1: Momentum factor.
            beta_2: Second momentum factor.
            epsilon: Small constant for numerical stability.
            exclude_from_weight_decay: List of regex patterns for variables to exclude.
            name: Optional name for the operations.
        """
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            name=name)
        
        self.weight_decay_rate = weight_decay_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay or []

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """Prepares local state for applying gradients.
        
        Migration changes:
        - Uses Keras-native state preparation
        - Better device handling
        - More efficient state management
        """
        super()._prepare_local(var_device, var_dtype, apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """Applies gradients to a variable.
        
        Migration changes:
        - Uses Keras resource variable operations
        - Better gradient application
        - Cleaner weight decay handling
        
        The update happens in 2 steps:
        1. Apply standard Adam update
        2. Apply weight decay separately if applicable
        """
        # Get the standard Adam update
        next_var = super()._resource_apply_dense(grad, var, apply_state)
        
        # Add weight decay if applicable
        if self._do_use_weight_decay(var.name):
            var.assign_sub(self.weight_decay_rate * var)
        
        return next_var

    def _do_use_weight_decay(self, param_name):
        """Determines whether to apply weight decay to a parameter.
        
        Migration changes:
        - Simplified regex matching
        - Better parameter name handling
        - Clearer logic
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Boolean indicating whether to apply weight decay
        """
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def get_config(self):
        """Returns the config of the optimizer.
        
        Migration changes:
        - Added proper serialization
        - Better config management
        - Supports model saving/loading
        
        Returns:
            Python dictionary of configuration
        """
        config = super().get_config()
        config.update({
            "weight_decay_rate": self.weight_decay_rate,
            "exclude_from_weight_decay": self.exclude_from_weight_decay,
        })
        return config


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, weight_decay_rate=0.01):
    """Creates an AdamW optimizer with learning rate schedule.
    
    Migration changes:
    - Uses Keras learning rate schedules
    - Removed TPU-specific code
    - Better schedule composition
    - Cleaner parameter handling
    
    The learning rate schedule includes:
    1. Linear warmup from 0 to init_lr in num_warmup_steps
    2. Linear decay from init_lr to 0 in remaining steps
    
    Args:
        init_lr: Initial learning rate after warmup
        num_train_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        weight_decay_rate: Weight decay rate for L2 regularization
    
    Returns:
        Configured AdamW optimizer instance
    """
    # Create learning rate schedule with linear warmup and linear decay
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps - num_warmup_steps,
        end_learning_rate=0.0,
        power=1.0  # Linear decay
    )

    if num_warmup_steps:
        learning_rate_fn = WarmupSchedule(
            initial_learning_rate=init_lr,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=num_warmup_steps)

    # Create and return the optimizer
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate_fn,
        weight_decay_rate=weight_decay_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    )
    
    return optimizer


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup.
    
    Migration changes:
    - Implemented as Keras schedule
    - Better rate calculation
    - Proper serialization
    - Cleaner logic
    
    The schedule:
    1. Linear increase from 0 to initial_rate in warmup_steps
    2. Follow decay_schedule_fn after warmup
    """

    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
        """Initializes the schedule.
        
        Args:
            initial_learning_rate: Target rate after warmup
            decay_schedule_fn: Schedule to use after warmup
            warmup_steps: Number of warmup steps
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        """Returns the learning rate for a given step.
        
        Migration changes:
        - Better step type handling
        - More efficient rate calculation
        - Cleaner conditional logic
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for this step
        """
        with tf.name_scope("WarmupSchedule"):
            # Convert to float
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)

            # Implement linear warmup
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

            # Get learning rate after warmup
            decay_learning_rate = self.decay_schedule_fn(step)

            # Return the proper learning rate
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: decay_learning_rate)

    def get_config(self):
        """Returns the config of the schedule.
        
        Migration changes:
        - Added proper serialization
        - Better config structure
        - Supports model saving
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
        }