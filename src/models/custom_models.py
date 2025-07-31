import keras
import tensorflow as tf
from src.models.custom_layers import L2Normalization, CosineLayer

@keras.saving.register_keras_serializable()
class VerificationModel(tf.keras.Model):
    """
    Modular Speaker Verification Model.

    Combines a backbone (feature extractor), an embedding projection, optional L2 normalization,
    and a cosine classification head (CosineLayer).

    Args:
        base_model (tf.keras.Model): Backbone model (e.g., ResNet18).
        number_of_classes (int): Number of speaker classes for classification.
        embedding_dim (int, optional): Size of embedding vector. Default: 512.
        return_embedding (bool, optional): If True, returns only embeddings (for verification); 
            if False, returns logits for classification. Default: False.
        base_training (bool, optional): If set, overrides 'training' flag for base model (controls BatchNorm, Dropout).
    """
    def __init__(
        self,
        base_model,
        number_of_classes,
        normalization_layer,
        cosine_layer,
        embedding_dim: int = 512,
        return_embedding: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        self.number_of_classes = number_of_classes
        self.return_embedding = return_embedding

        self.embedding_layer = tf.keras.layers.Dense(
            embedding_dim,
            activation='tanh',
            use_bias=True,
            name='embedding_dense'
        )
        self.normalization_layer = normalization_layer
        self.cosine_layer = cosine_layer

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Input tensor (e.g., spectrograms).
            training (bool, optional): Training mode (Keras convention).
        Returns:
            Embeddings (if return_embedding=True) or logits for classification.
        """

        x = self.base_model(inputs, training=training)
        x = self.embedding_layer(x)
        x = self.normalization_layer(x)
        if self.return_embedding:
            return x
        return self.cosine_layer(x)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "base_model": keras.saving.serialize_keras_object(self.base_model),
            "normalization_layer": keras.saving.serialize_keras_object(
                self.normalization_layer
            ),
            "cosine_layer": keras.saving.serialize_keras_object(self.cosine_layer),
            "number_of_classes": self.number_of_classes,
            "embedding_dim": self.embedding_dim,
            "return_embedding": self.return_embedding
        }

    @classmethod
    def from_config(cls, config):
        base_model = keras.saving.deserialize_keras_object(config.pop("base_model"))
        normalization_layer = keras.saving.deserialize_keras_object(config.pop("normalization_layer"))
        cosine_layer = keras.saving.deserialize_keras_object(config.pop("cosine_layer"))
        return cls(base_model=base_model,
                   normalization_layer=normalization_layer,
                   cosine_layer=cosine_layer,
                   **config)
