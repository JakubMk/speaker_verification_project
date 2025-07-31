import numpy as np
from tqdm import tqdm
import pandas as pd
from src.data import test_dataset
from src.models import VerificationModel

def eer(model: VerificationModel, test_df: pd.DataFrame, batch_size: int = 16) -> float:
    """
    Calculates the Equal Error Rate (EER) on the validation data.

    Args:
        model (VerificationModel): Trained model returning embeddings (must have attribute `return_embedding`)
        test_df (pd.DataFrame): DataFrame for validation, with columns ['speaker_1', 'speaker_2', 'y_true']
        batch_size (int): Batch size for inference

    Returns:
        float: EER value (Equal Error Rate, lower is better)

    Example:
        eer_score = eer(model, test_df, batch_size=32)
    """
    test_df = test_df.copy()
    model.return_embedding = True  # Enable embeddings output

    # Get all embeddings for the test set
    test_ds, all_test_paths = test_dataset(test_df, batch_size=batch_size)
    y_pred_model = model.predict(test_ds, batch_size=batch_size)

    # Map file paths to embeddings
    embeddings_dict = {path: emb for (path, emb) in zip(all_test_paths, y_pred_model)}

    # Compute cosine similarity for each pair
    def get_cosine_similarity(row):
        emb1 = embeddings_dict[row['speaker_1']]
        emb2 = embeddings_dict[row['speaker_2']]
        return np.dot(emb1, emb2)

    test_df['cosine_similarity'] = test_df.apply(get_cosine_similarity, axis=1)

    margins = np.linspace(-0.99, 0.99, 2000)
    y_true = test_df.y_true.values
    
    results = []
    total_negatives = (y_true == 0).sum()
    total_positives = (y_true == 1).sum()

    for margin in tqdm(margins, desc="EER Threshold Search"):
        y_pred = (test_df['cosine_similarity'] >= margin).astype(int)

        false_accepts = ((y_true == 0) & (y_pred == 1)).sum()
        far = false_accepts / total_negatives if total_negatives > 0 else 0

        false_rejects = ((y_true == 1) & (y_pred == 0)).sum()
        frr = false_rejects / total_positives if total_positives > 0 else 0

        results.append({
            'margin': margin,
            'FAR': far,
            'FRR': frr,
            'diff': np.abs(far - frr)
        })

    model.return_embedding = False  # Reset to default

    eer_df = pd.DataFrame(results)
    eer_row = eer_df.loc[eer_df['diff'].idxmin()]

    return eer_row['FAR']  # Equal Error Rate at the optimal threshold
