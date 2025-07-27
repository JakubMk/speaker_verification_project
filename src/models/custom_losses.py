import tensorflow as tf
import keras
import numpy as np
from src.utils import prepare_masks

@keras.saving.register_keras_serializable()
class AdaCosLoss(tf.keras.losses.Loss):
    """
    Adaptive Cosine Loss (AdaCos).

    Implements the AdaCos loss function as described in:
    "AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations"
    (Zhang et al., 2019).

    Args:
        num_classes (int): Number of classes in the classification problem.
        name (str, optional): Name for the loss instance.
    """
    def __init__(self, num_classes=None, name="AdaCos", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.scale = tf.Variable(
            np.sqrt(2) * np.log(num_classes - 1), 
            dtype=tf.float32, trainable=False
        )

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: (batch_size,) integer labels [0, num_classes-1].
            y_pred: (batch_size, num_classes) classification cosine similarities.

        Returns:
            Tensor scalar: Mean AdaCos loss over the batch.
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(
            y_pred, 
            -1.0 + tf.keras.backend.epsilon(), 
            1.0 - tf.keras.backend.epsilon()
        )
        # correct class mask
        mask = tf.one_hot(y_true, depth=self.num_classes) # shape (batch_size, n_classes)
        # get theta angles for corresponding class
        theta_true = tf.math.acos(tf.boolean_mask(y_pred, mask)) # shape (batch_size,)
        # compute median of 'correct' angles
        theta_med = tf.keras.ops.median(theta_true)
        # get non-corresponding cosine values (cos(theta) j is not yi)
        neg_mask = tf.logical_not(mask > 0) # shape (batch_size, n_classes)
        cos_theta_neg = tf.boolean_mask(y_pred, neg_mask) # shape (batch_size*(n_classes-1),)

        neg_y_pred = tf.reshape(cos_theta_neg, [-1, self.num_classes - 1]) # shape (batch_size, n_classes-1)
        
        B_avg = tf.reduce_mean(tf.reduce_sum(tf.math.exp(self.scale * neg_y_pred), axis=-1))
        #B_avg = tf.cast(B_avg, tf.float32)

        #with tf.control_dependencies([theta_med, B_avg]):
        new_scale = (
            tf.math.log(B_avg) / 
            tf.math.cos(tf.minimum(tf.constant(np.pi / 4), theta_med))
        )
        # keep current scale if new_scale is invalid
        safe_scale = tf.cond(
            tf.math.is_finite(new_scale) & (new_scale > 0),
            lambda: new_scale,
            lambda: self.scale
        )
        self.scale.assign(safe_scale)
        logits = self.scale * y_pred
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        
        return tf.reduce_mean(loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'num_classes': self.num_classes}
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(num_classes={self.num_classes}, "
                f"name='{self.name}')")

    def __str__(self):
        return self.__repr__()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        if not isinstance(value, int):
            raise TypeError(f"`num_classes` must be an int, got {type(value).__name__}")
        if value < 2:
            raise ValueError(f"`num_classes` must be >= 2, got {value}")
        self._num_classes = value

@keras.saving.register_keras_serializable()
class AdaCosLossMargin(tf.keras.losses.Loss):
    """
    Adaptive Cosine Loss with Margin (AdaCosMargin).

    Extends AdaCos by introducing a fixed margin penalty for the target class logits, 
    encouraging greater separation between classes in angular (cosine) space.

    Reference:
    - AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations (Zhang et al., 2019)
    - Large Margin Cosine Loss (CosFace): https://arxiv.org/abs/1801.09414

    Args:
        margin (float): Margin to subtract from the target class cosine similarity (0.0–1.0).
        num_classes (int): Number of classes.
        name (str, optional): Name for the loss.
    """
    def __init__(self, margin=0.1, num_classes=None, name="AdaCosLossMargin", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin
        self.num_classes = num_classes
        self.scale = tf.Variable(
            np.sqrt(2) * np.log(num_classes - 1), 
            dtype=tf.float32, trainable=False
        )

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: (batch_size,) integer labels [0, num_classes-1].
            y_pred: (batch_size, num_classes) cosine similarities.

        Returns:
            Tensor scalar: Mean AdaCosMargin loss over the batch.
        """
        batch_size = tf.shape(y_pred)[0]
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(
            y_pred,
            -1.0 + tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )
        mask = tf.one_hot(y_true, depth=self.num_classes)
        theta_true = tf.math.acos(tf.boolean_mask(y_pred, mask))
        theta_med = tf.keras.ops.median(theta_true)
        neg_mask = tf.cast(tf.logical_not(mask > 0), dtype=tf.float32)
        cos_theta_neg = tf.boolean_mask(y_pred, neg_mask)
        neg_y_pred = tf.reshape(cos_theta_neg, [batch_size, self.num_classes - 1])
        B_avg = tf.reduce_mean(tf.reduce_sum(tf.math.exp(self.scale * neg_y_pred), axis=-1))
        B_avg = tf.cast(B_avg, tf.float32)

        with tf.control_dependencies([theta_med, B_avg]):
            new_scale = (
                tf.math.log(B_avg) /
                tf.math.cos(tf.minimum(tf.constant(np.pi / 4), theta_med))
            )
            safe_scale = tf.cond(
                tf.math.is_finite(new_scale) & (new_scale > 0),
                lambda: new_scale,
                lambda: self.scale
            )
            self.scale.assign(safe_scale)
            logits = self.scale * (y_pred - self.margin * mask)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        return tf.reduce_mean(loss)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'num_classes': self.num_classes,
            'margin': self.margin
        }
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(margin={self.margin}, num_classes={self.num_classes}, "
                f"name='{self.name}')")

    def __str__(self):
        return self.__repr__()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        if not isinstance(value, int):
            raise TypeError(f"`num_classes` must be an int, got {type(value).__name__}")
        if value < 2:
            raise ValueError(f"`num_classes` must be >= 2, got {value}")
        self._num_classes = value

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"`margin` must be a float or int, got {type(value).__name__}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"`margin` must be between 0.0 and 1.0, got {value}")
        self._margin = value

@keras.saving.register_keras_serializable()
class LMCLoss(tf.keras.losses.Loss):
    """
    Large Margin Cosine Loss (LMCLoss), also known as CosFace loss.
    
    Encourages larger angular margins between classes by modifying the cosine similarity logits.
    Refer to: [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
    
    Args:
        margin (float): Margin value to subtract from target class cosine. Must be between 0.0 and 1.0.
        scale (float): Scaling factor for logits. Must be positive.
        name (str): Optional name for the loss.
    """
    def __init__(self, margin=0.35, scale=64.0, name="LMCLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin
        self.scale = scale

    def call(self, y_true, y_pred):
        """
        Computes the LMCLoss.

        Args:
            y_true: Tensor of shape (batch_size,), integer class labels.
            y_pred: Tensor of shape (batch_size, num_classes), cosine similarity logits.

        Returns:
            Tensor: Scalar mean loss value.
        """
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[1]
        # correct class mask (margin subtraction)
        mask = tf.one_hot(y_true, depth=num_classes)
        logits = self.scale * (y_pred - self.margin * mask)
        # sparse softmax cross-entropy per sample
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        return tf.reduce_mean(loss)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "margin": self.margin,
            "scale": self.scale
        }

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"`margin` must be a float or int, got {type(value).__name__}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"`margin` must be between 0.0 and 1.0, got {value}")
        self._margin = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"`scale` must be a float or int, got {type(value).__name__}")
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"`scale` must be positive, got {value}")
        self._scale = value

    def __repr__(self):
        return (f"{self.__class__.__name__}(margin={self.margin}, scale={self.scale}, "
                f"name='{self.name}')")

    def __str__(self):
        return self.__repr__()
    
@keras.saving.register_keras_serializable()
class ContrastiveLoss(tf.keras.losses.Loss):
    """
    Custom contrastive loss for speaker verification using cosine similarity.

    Penalizes pairs based on cosine similarity margin:
      - Positive pairs (same class): loss if similarity < margin - alpha_positive
      - Negative pairs (different class): loss if similarity > margin - alpha_negative

    References:
      - "Siamese Neural Networks for One-shot Image Recognition"
      - "FaceNet: A Unified Embedding for Face Recognition and Clustering"

    Args:
        margin (float): Cosine similarity margin (0.0 - 1.0).
        alpha_positive (float): Additive slack for positive loss (default: 0.05).
        alpha_negative (float): Additive slack for negative loss (default: 0.1).
        name (str): Name for the loss instance.
        reduction: Reduction type (see tf.keras.losses.Loss).
    """
    def __init__(
        self,
        margin: float = 0.8,
        alpha_positive: float = 0.05,
        alpha_negative: float = 0.1,
        name="custom_contrastive_loss",
        reduction=None,
        **kwargs
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.margin = margin
        self.alpha_positive = alpha_positive
        self.alpha_negative = alpha_negative

    def call(self, y_true, y_pred):
        """
        Computes the contrastive loss for a batch.

        Args:
            y_true: (batch_size,) integer labels [0, num_classes-1].
            y_pred: (batch_size, embedding_dim) L2-normalized embeddings.

        Returns:
            Tensor: Scalar loss value.
        """
        return self.contrastive_loss(
            y_pred, y_true,
            margin=self.margin,
            alpha_positive=self.alpha_positive,
            alpha_negative=self.alpha_negative
        )

    def contrastive_loss(
        self, batch_embedding, batch_labels, margin=0.8, alpha_positive=0.05, alpha_negative=0.1
    ):
        """
        Internal loss calculation, expects L2-normalized embeddings.

        Args:
            batch_embedding: (batch_size, embedding_dim) tensor.
            batch_labels: (batch_size,) tensor of int.
            margin, alpha_positive, alpha_negative: see above.

        Returns:
            Scalar Tensor loss.
        """
        cs_mtx, pos_mask, neg_mask = prepare_masks(batch_embedding, batch_labels)

        # Positive and negative cosine similarities
        pos_cs_sim = tf.boolean_mask(cs_mtx, pos_mask)
        neg_cs_sim = tf.boolean_mask(cs_mtx, neg_mask)

        # Positive: penalize if similarity < margin - alpha
        Ls = tf.reduce_sum(tf.maximum(0.0, margin - pos_cs_sim + alpha_positive)) / tf.reduce_sum(tf.cast(pos_mask, tf.float32))
        # Negative: penalize if similarity > margin - alpha
        Ld = tf.reduce_sum(tf.maximum(0.0, neg_cs_sim - margin + alpha_negative)) / tf.reduce_sum(tf.cast(neg_mask, tf.float32))
        return Ls + Ld

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
            "margin": self.margin,
            "alpha_positive": self.alpha_positive,
            "alpha_negative": self.alpha_negative
            }

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"`margin` must be a float, got {type(value).__name__}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"`margin` must be between 0.0 and 1.0, got {value}")
        self._margin = value

    @property
    def alpha_positive(self):
        return self._alpha_positive

    @alpha_positive.setter
    def alpha_positive(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"`alpha_positive` must be a float, got {type(value).__name__}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"`alpha_positive` must be between 0.0 and 1.0, got {value}")
        self._alpha_positive = value

    @property
    def alpha_negative(self):
        return self._alpha_negative

    @alpha_negative.setter
    def alpha_negative(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"`alpha_negative` must be a float, got {type(value).__name__}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"`alpha_negative` must be between 0.0 and 1.0, got {value}")
        self._alpha_negative = value

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(margin={self.margin}, "
            f"alpha_positive={self.alpha_positive}, "
            f"alpha_negative={self.alpha_negative}, name='{self.name}')"
        )

    def __str__(self):
        return self.__repr__()
