import tensorflow as tf
from src.utils import contrastive_accuracy

@tf.keras.utils.register_keras_serializable()
class ContrastiveAccuracy(tf.keras.metrics.Metric):
    """
    Computes accuracy, precision, and recall for a contrastive (pair-based) speaker verification setup
    using cosine similarity and a user-defined margin.

    For each batch, compares all possible pairs in the batch (excluding self-similarity),
    then accumulates counts of true positives, true negatives, predicted positives, and total pairs.

    Returns a dictionary with the following keys:
        - '!accuracy':  (TP + TN) / All pairs
        - '!precision': TP / All predicted positives
        - '!recall':    TP / All positive pairs

    Args:
        margin (float): Cosine similarity threshold for positive/negative decision (default 0.8).
        name (str): Name for the metric.
    """
    def __init__(self, margin: float = 0.8, name: str = "contrastive_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin
        self.true_pos_count = self.add_weight(name="true_pos_count", initializer="zeros") # y_pred=1 & y_true=1
        self.true_neg_count = self.add_weight(name="true_neg_count", initializer="zeros") # y_pred=0 & y_true=0
        self.pred_pos_count = self.add_weight(name="pred_pos_count", initializer="zeros") # sum(y_pred=1)
        self.all_pos_label_count = self.add_weight(name="all_pos_label_count", initializer="zeros") # sum(y_true=1)
        self.all_label_count = self.add_weight(name="all_label_count", initializer="zeros") # no. of pairs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates internal counts for accuracy/precision/recall computation.
        Assumes y_pred: embedding vectors, y_true: integer labels.
        """
        true_pos_count, true_neg_count, pred_pos_count, all_pos_label_count, all_label_count = contrastive_accuracy(
            y_pred, y_true, margin=self.margin
        )
        self.true_pos_count.assign_add(true_pos_count)
        self.true_neg_count.assign_add(true_neg_count)
        self.pred_pos_count.assign_add(pred_pos_count)
        self.all_pos_label_count.assign_add(all_pos_label_count)
        self.all_label_count.assign_add(all_label_count)

    def result(self):
        """
        Returns a dictionary with accuracy, precision, and recall.
        Uses tf.math.divide_no_nan to avoid NaN when denominators are zero.
        """
        return {
            '!accuracy': tf.math.divide_no_nan(self.true_pos_count + self.true_neg_count, self.all_label_count),
            '!precision': tf.math.divide_no_nan(self.true_pos_count, self.pred_pos_count),
            '!recall': tf.math.divide_no_nan(self.true_pos_count, self.all_pos_label_count)
        }

    def reset_state(self):
        """Resets all stateful counters to zero at the start of each epoch."""
        self.true_pos_count.assign(0.0)
        self.true_neg_count.assign(0.0)
        self.pred_pos_count.assign(0.0)
        self.all_pos_label_count.assign(0.0)
        self.all_label_count.assign(0.0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "margin": self.margin}

    @property
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def margin(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"`margin` must be a float, got {type(value).__name__}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"`margin` must be between 0.0 and 1.0, got {value}")
        self._margin = value
