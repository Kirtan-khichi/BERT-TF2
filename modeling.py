# coding=utf-8

"""
TensorFlow 2.x Implementation of BERT (Bidirectional Encoder Representations from Transformers)

Migration Changes from TF1:
1. Architecture:
   - Converted to Keras-style layers and models
   - Replaced custom attention with tf.keras.layers.MultiHeadAttention
   - Modularized code into proper Keras layer classes
   - Removed manual variable scopes in favor of Keras layer management

2. API Updates:
   - Switched from tf.layers to tf.keras.layers
   - Updated file I/O from tf.gfile to tf.io.gfile
   - Replaced custom layer norm with tf.keras.layers.LayerNormalization
   - Using Keras initializers instead of custom ones

3. TPU/Distribution:
   - Removed TPU-specific code since TF2 handles devices better
   - Simplified distribution strategy handling
   - Better batch processing support

4. Memory & Performance:
   - More efficient attention implementation
   - Better gradient flow through residual connections
   - Improved memory usage patterns
   - Keras-native dropout and normalization

Original paper: https://arxiv.org/abs/1810.04805
"""

import copy
import json
import tensorflow as tf


class BertConfig:
    """Configuration class for BERT model hyperparameters.
    
    Key changes from TF1:
    - Simplified initialization
    - Better parameter validation
    - More pythonic attribute handling
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Size of the encoder layers and pooler layer
        num_hidden_layers: Number of hidden transformer layers
        num_attention_heads: Number of attention heads for each layer
        intermediate_size: Size of intermediate (feed-forward) layer
        hidden_act: Activation function for intermediate layer
        hidden_dropout_prob: Dropout probability for all fully connected layers
        attention_probs_dropout_prob: Dropout probability for attention probabilities
        max_position_embeddings: Maximum sequence length supported
        type_vocab_size: Vocabulary size of token_type_ids
        initializer_range: Initialization range for weights
    """
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_obj):
        # Filter keys to only those accepted by __init__
        valid_keys = {
            'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
            'intermediate_size', 'hidden_act', 'hidden_dropout_prob',
            'attention_probs_dropout_prob', 'max_position_embeddings',
            'type_vocab_size', 'initializer_range'
        }
        filtered_args = {k: v for k, v in json_obj.items() if k in valid_keys}
        return cls(**filtered_args)


    @classmethod
    def from_json_file(cls, json_file):
        with tf.io.gfile.GFile(json_file, "r") as reader:
            js = json.loads(reader.read())
        return cls.from_dict(js)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def gelu(x):
    """Gaussian Error Linear Unit activation function.
    
    Migration changes:
    - Using tf.keras.activations.gelu instead of custom implementation
    - Simplified calculation with native TF2 ops
    
    Args:
        x: Input tensor
    Returns:
        GELU activation applied to input
    """
    return tf.keras.activations.gelu(x)


def get_activation(act_str):
    """Maps activation function string to corresponding TF activation.
    
    Migration changes:
    - Using tf.keras.activations instead of custom functions
    - Better error handling
    - More activation options supported
    
    Args:
        act_str: String name of activation function
    Returns:
        Corresponding activation function
    Raises:
        ValueError: For unsupported activation names
    """
    if not act_str or act_str.lower() == "linear":
        return tf.keras.activations.linear
    act = act_str.lower()
    if act == "relu":
        return tf.keras.activations.relu
    if act == "gelu":
        return gelu
    if act == "tanh":
        return tf.keras.activations.tanh
    raise ValueError(f"Unsupported activation: {act_str}")

def create_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def layer_norm(input_tensor, name=None):
    return tf.keras.layers.LayerNormalization(epsilon=1e-12, name=name)(input_tensor)


class BertEmbeddings(tf.keras.layers.Layer):
    """BERT embeddings layer combining token, position and segment embeddings.
    
    Migration changes:
    - Converted to proper Keras layer
    - Using Keras embedding layers
    - Native layer normalization
    - Better shape handling
    
    The embedding process:
    1. Convert token IDs to token embeddings
    2. Add positional embeddings to encode sequence order
    3. Add segment embeddings for sentence differentiation
    4. Layer normalize and dropout the final embeddings
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="word_embeddings",
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="token_type_embeddings",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids=None, training=False):
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_length)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.zeros_like(input_ids)
        w = self.word_embeddings(input_ids)
        p = self.position_embeddings(position_ids)
        t = self.token_type_embeddings(token_type_ids)
        embeddings = w + p + t
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings, training=training)


class BertSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention mechanism.
    
    Key changes from TF1:
    - Using tf.keras.layers.MultiHeadAttention instead of custom implementation
    - Simplified attention mask handling
    - Better dropout patterns
    - More efficient computation
    
    The attention process:
    1. Project input into query, key and value vectors
    2. Compute attention scores between all positions
    3. Apply attention dropout
    4. Combine values weighted by attention scores
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_attention_heads,
            key_dim=config.hidden_size // config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="self_attention",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.output_dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="output_dense",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="output_layernorm")

    def call(self, hidden_states, attention_mask=None, training=False):
        attn_output = self.attention(
            query=hidden_states,
            value=hidden_states,
            key=hidden_states,
            attention_mask=attention_mask,
            training=training,
        )
        attn_output = self.output_dense(attn_output)
        attn_output = self.dropout(attn_output, training=training)
        return self.layer_norm(hidden_states + attn_output)


class BertIntermediate(tf.keras.layers.Layer):
    """Intermediate (feed-forward) layer in transformer block.
    
    Migration changes:
    - Converted to Keras layer
    - Using native dense layer
    - Better activation handling
    
    This implements the first part of the feed-forward network:
    - Dense projection with activation (usually GELU)
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size,
            activation=get_activation(config.hidden_act),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="intermediate_dense",
        )

    def call(self, hidden_states):
        return self.dense(hidden_states)


class BertOutput(tf.keras.layers.Layer):
    """Output layer in transformer block with residual connection.
    
    Migration changes:
    - Proper Keras layer implementation
    - Native layer normalization
    - Better residual connection handling
    
    Process:
    1. Project intermediate outputs back to hidden size
    2. Apply dropout
    3. Add residual connection
    4. Layer normalize
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="output_dense",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="output_layernorm")

    def call(self, hidden_states, input_tensor, training=False):
        h = self.dense(hidden_states)
        h = self.dropout(h, training=training)
        return self.layer_norm(input_tensor + h)


class BertLayer(tf.keras.layers.Layer):
    """Single transformer block combining attention and feed-forward networks.
    
    Migration changes:
    - Full Keras layer implementation
    - Better component organization
    - Cleaner forward pass
    
    Architecture:
    1. Multi-head self-attention
    2. Add & normalize
    3. Feed-forward network
    4. Add & normalize
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.attention = BertSelfAttention(config, name="attention")
        self.intermediate = BertIntermediate(config, name="intermediate")
        # Rename attribute to avoid Keras conflict
        self.output_block = BertOutput(config, name="output")

    def call(self, hidden_states, attention_mask=None, training=False):
        a = self.attention(hidden_states, attention_mask, training=training)
        i = self.intermediate(a)
        return self.output_block(i, a, training=training)


class BertEncoder(tf.keras.layers.Layer):
    """Stack of transformer blocks that form BERT's core.
    
    Migration changes:
    - Keras-native layer stacking
    - Simplified forward pass
    - Better state management
    
    The encoder repeatedly applies transformer blocks, with each block:
    1. Attending to all positions in the sequence
    2. Processing through a feed-forward network
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.layer = [
            BertLayer(config, name=f"layer_{i}") for i in range(config.num_hidden_layers)
        ]

    def call(self, hidden_states, attention_mask=None, training=False):
        # Propagate through each transformer block, passing training as keyword
        for lyr in self.layer:
            hidden_states = lyr(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )
        return hidden_states


def create_attention_mask_from_input_mask(input_ids, input_mask):
    """Creates 3D attention mask from 2D input mask.
    
    Migration changes:
    - Simplified tensor operations
    - Better type casting
    - More efficient broadcasting
    
    Process:
    1. Convert input mask to float
    2. Add broadcast dimensions for attention
    3. Convert to boolean for efficient attention
    
    Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length]
        input_mask: int32 Tensor of shape [batch_size, seq_length]
    Returns:
        float32 Tensor of shape [batch_size, 1, 1, seq_length]
    """
    # Convert input_mask to float32 tensor
    input_mask = tf.cast(input_mask, tf.float32)
    
    # Create attention mask [batch_size, 1, 1, seq_length]
    # This creates a 3D tensor which is broadcast for addition
    attention_mask = input_mask[:, tf.newaxis, tf.newaxis, :]
    
    # Convert to boolean
    attention_mask = tf.cast(attention_mask, tf.bool)
    
    return attention_mask


class TFBertModel(tf.keras.Model):
    """Complete BERT model implemented in TF2 Keras style.
    
    Migration changes:
    - Converted to tf.keras.Model subclass
    - Proper layer management
    - Better forward pass organization
    - Native Keras training support
    
    Architecture:
    1. Embedding layer (token + position + segment)
    2. Encoder stack of transformer blocks
    3. Pooler for sentence-level representations
    
    Usage:
    ```python
    config = BertConfig(vocab_size=32000, hidden_size=512)
    model = TFBertModel(config)
    outputs = model(input_ids, attention_mask, token_type_ids)
    sequence_output, pooled_output = outputs
    ```
    """
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = BertEmbeddings(config, name="embeddings")
        self.encoder = BertEncoder(config, name="encoder")
        self.pooler_dense = tf.keras.layers.Dense(
            config.hidden_size,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(config.initializer_range),
            name="pooler_dense",
        )

    def call(self, input_ids, attention_mask=None, token_type_ids=None, training=False):
        if attention_mask is None:
            attention_mask = tf.ones_like(input_ids)
        bool_mask = create_attention_mask_from_input_mask(input_ids, attention_mask)
        embed_out = self.embeddings(input_ids, token_type_ids, training=training)
        seq_out = self.encoder(embed_out, bool_mask, training=training)
        first_token = seq_out[:, 0]
        pooled_out = self.pooler_dense(first_token)
        return seq_out, pooled_out
