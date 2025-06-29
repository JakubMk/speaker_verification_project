import tensorflow as tf

def prepare_masks(batch_embedding: tf.Tensor, batch_labels: tf.Tensor):
    """
    Computes the cosine similarity matrix and boolean masks for positive and negative pairs in a batch.

    Args:
        batch_embedding (tf.Tensor): Embeddings for the batch, shape (batch_size, embedding_dim).
        batch_labels (tf.Tensor): Class labels for the batch, shape (batch_size,).

    Returns:
        tuple:
            cs_mtx (tf.Tensor): Cosine similarity matrix (batch_size, batch_size).
            pos_mask (tf.Tensor): Boolean mask of positive pairs (True where pair is positive, off-diagonal).
            neg_mask (tf.Tensor): Boolean mask of negative pairs (True where pair is negative, off-diagonal).
    """
    batch_size = tf.shape(batch_embedding)[0]

    # Cosine similarity matrix: (batch_size, batch_size)
    cs_mtx = tf.linalg.matmul(batch_embedding, batch_embedding, transpose_a=False, transpose_b=True)
    
    # Reshape labels to (batch_size, 1) for broadcasting
    label_batch = tf.reshape(batch_labels, (-1, 1))

    # Boolean label matrix: True for same class (positive pair)
    label_mtx = tf.equal(label_batch, tf.transpose(label_batch))

    # Remove self-similarity (diagonal)
    diag_mask = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))

    # Positive pairs: same class, not on diagonal
    pos_mask = tf.logical_and(label_mtx, diag_mask)
    # Negative pairs: different class, not on diagonal
    neg_mask = tf.logical_and(tf.logical_not(label_mtx), diag_mask)
    return cs_mtx, pos_mask, neg_mask

def contrastive_loss(
    batch_embedding: tf.Tensor, 
    batch_labels: tf.Tensor, 
    margin: float = 0.8, 
    alpha_positive: float = 0.05, 
    alpha_negative: float = 0.1
) -> tf.Tensor:
    """
    Computes contrastive loss using cosine similarity for a batch.

    Args:
        batch_embedding (tf.Tensor): Normalized embeddings (batch_size, embedding_dim).
        batch_labels (tf.Tensor): Class labels (batch_size,).
        margin (float): Cosine similarity margin separating positive and negative pairs.
        alpha_positive (float): Smoothing/additional margin for positive pairs.
        alpha_negative (float): Smoothing/additional margin for negative pairs.

    Returns:
        tf.Tensor: Scalar loss value for the batch.
    """
    cs_mtx, pos_mask, neg_mask = prepare_masks(batch_embedding, batch_labels)

    # Extract cosine similarities for positive and negative pairs
    pos_cs_sim = tf.boolean_mask(cs_mtx, pos_mask)
    neg_cs_sim = tf.boolean_mask(cs_mtx, neg_mask)

    # Positive loss: margin - sim + alpha_positive
    Ls = tf.reduce_sum(tf.maximum(0.0, margin - pos_cs_sim + alpha_positive)) / tf.reduce_sum(tf.cast(pos_mask, tf.float32))
    # Negative loss: sim - margin + alpha_negative
    Ld = tf.reduce_sum(tf.maximum(0.0, neg_cs_sim - margin + alpha_negative)) / tf.reduce_sum(tf.cast(neg_mask, tf.float32))
    
    return Ls + Ld

def contrastive_accuracy(
    batch_embedding: tf.Tensor, 
    batch_labels: tf.Tensor, 
    margin: float = 0.8
) -> tuple:
    """
    Calculates true positive and true negative counts for contrastive prediction at a fixed margin.

    Args:
        batch_embedding (tf.Tensor): Embedding matrix, shape (batch_size, embedding_dim).
        batch_labels (tf.Tensor): Class labels, shape (batch_size,).
        margin (float): Cosine similarity margin threshold.

    Returns:
        tuple:
            true_pos_count (tf.Tensor): Correct positive predictions count.
            true_neg_count (tf.Tensor): Correct negative predictions count.
            pred_pos_count (tf.Tensor): Total positive predictions.
            all_pos_label_count (tf.Tensor): Total positive label pairs.
            all_label_count (tf.Tensor): Total possible pairs (off-diagonal).
    """
    batch_size = tf.shape(batch_labels)[0]
    diag_mask = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
    
    cs_mtx, pos_label_mask, neg_label_mask = prepare_masks(batch_embedding, batch_labels)

    # y_true=1 and all y_true counts
    all_pos_label_count = tf.reduce_sum(tf.cast(pos_label_mask, tf.float32))
    all_label_count = all_pos_label_count + tf.reduce_sum(tf.cast(neg_label_mask, tf.float32))
    
    # Prediction masks based on margin threshold
    pred_pos_mtx = tf.logical_and(cs_mtx >= margin, diag_mask)
    pred_neg_mtx = tf.logical_and(cs_mtx < margin, diag_mask)
    pred_pos_count = tf.reduce_sum(tf.cast(pred_pos_mtx, tf.float32))

    # prediction mask (model: cosine > margin → positive)
    true_pos_pred_mask = tf.logical_and(pred_pos_mtx, pos_label_mask)
    true_neg_pred_mask = tf.logical_and(pred_neg_mtx, neg_label_mask)

    # Correct prediction counts
    true_pos_count = tf.reduce_sum(tf.cast(true_pos_pred_mask, tf.float32))
    true_neg_count = tf.reduce_sum(tf.cast(true_neg_pred_mask, tf.float32))
    
    return true_pos_count, true_neg_count, pred_pos_count, all_pos_label_count, all_label_count
