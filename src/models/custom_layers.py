import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class L2Normalization(tf.keras.layers.Layer):
    """
    Applies L2 normalization to the last axis of the input tensor.
    
    This is used as a top layer in speaker embedding models before
    cosine similarity computation.
    """
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape

@keras.saving.register_keras_serializable()
class CosineLayer(tf.keras.layers.Layer):
    """
    Dense layer with L2-normalized weights, for cosine similarity-based classification.

    Args:
        out_features (int): Number of output features/classes.
        use_bias (bool): Whether to use bias term.
        name (str, optional): Layer name.
    """
    def __init__(self, out_features, use_bias=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_features = out_features
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.out_features),
            initializer='glorot_uniform',
            trainable=True,
            name='weights'
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.out_features,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        else:
            self.b = None
        super().build(input_shape)
        
    def call(self, inputs):
        w_normalized = tf.math.l2_normalize(self.w, axis=0)
        logits = tf.linalg.matmul(inputs, w_normalized)
        if self.use_bias:
            logits = logits + self.b
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'out_features': self.out_features,
            'use_bias': self.use_bias
        }
