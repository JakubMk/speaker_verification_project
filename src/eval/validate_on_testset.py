import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data import test_dataset

def validate_on_testset(model, test_df: pd.DataFrame, batch_size: int = 16):
    """
    Evaluates a speaker verification model on the provided test set using cosine similarity and computes EER.

    Args:
        model: Trained Verification Model that returns normalized embeddings for audio files.
        test_df (pd.DataFrame): DataFrame with columns ['speaker_1', 'speaker_2', 'y_true'].
        batch_size (int): Batch size for model prediction.

    Returns:
        eer_df (pd.DataFrame): FAR, FRR and diff for each margin.
        test_df (pd.DataFrame): Test DataFrame with added cosine_similarity column.
        eer_row (pd.Series): Row of eer_df with threshold closest to equal FAR and FRR.
    """
    test_df = test_df.copy()
    model.return_embedding = True

    # test_dataset returns a tf.data.Dataset and array of all unique test file paths
    test_ds, all_test_paths = test_dataset(test_df, batch_size=batch_size)
    y_pred_model = model.predict(test_ds, batch_size=batch_size)

    # create a dictionary mapping file paths to their predicted embeddings
    embeddings_dict = {path: emb for path, emb in zip(all_test_paths, y_pred_model)}

    def get_cosine_similarity(row):
        emb1 = embeddings_dict[row['speaker_1']]
        emb2 = embeddings_dict[row['speaker_2']]
        return np.dot(emb1, emb2)

    # Apply the cosine similarity function to each row in the test DataFrame
    test_df['cosine_similarity'] = test_df.progress_apply(get_cosine_similarity, axis=1)

    margins = np.linspace(-0.99, 0.99, 2000)
    y_true = test_df['y_true']

    results = []
    total_negatives = (y_true == 0).sum()
    total_positives = (y_true == 1).sum()

    for margin in tqdm(margins, desc="EER Threshold Search"):
        y_pred = (test_df['cosine_similarity'] >= margin).astype(int)
        false_accepts = ((y_true == 0) & (y_pred == 1)).sum()
        far = false_accepts / total_negatives if total_negatives > 0 else 0
        false_rejects = ((y_true == 1) & (y_pred == 0)).sum()
        frr = false_rejects / total_positives if total_positives > 0 else 0
        results.append({'margin': margin, 'FAR': far, 'FRR': frr, 'diff': np.abs(far-frr)})

    model.return_embedding = False

    eer_df = pd.DataFrame(results)
    eer_row = eer_df.loc[eer_df['diff'].idxmin()]
    
    return eer_df, test_df, eer_row
