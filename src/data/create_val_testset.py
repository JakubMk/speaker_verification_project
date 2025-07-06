import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random
from src.data import sample_diff_pos_utts
from src.data import test_dataset

def create_val_testset(
    model,
    df: pd.DataFrame,
    save_path: str | Path,
    n: int = 5,
    n_candidates: int = 50,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Generates a validation/test set for speaker verification with hard positive and negative pairs.
   
    For each speaker in df:
      - Generates n_candidates positive and n_candidates negative candidate pairs.
      - Computes cosine similarity for each pair using the provided model.
      - Selects the n_pairs lowest-similarity positive pairs and n_pairs highest-similarity negative pairs.
      - Alternates them in the output (pos, neg, pos, neg, ...).
   
    Args:
        model: Trained speaker verification model capable of producing embeddings (expects spectrogram input).
        df (pd.DataFrame): DataFrame with columns ['id', 'utt', 'file_path'] for each audio sample.
        save_path (str): Path to save the resulting CSV (no index).
        n_candidates (int): Number of candidate positive and negative pairs to generate per speaker.
        n_pairs (int): Number of hardest positive and negative pairs to select per speaker for the final set.
        batch_size (int): Batch size for model prediction (default 16).
   
    Returns:
        pd.DataFrame: Final DataFrame with columns ['speaker_1', 'speaker_2', 'y_true'] containing selected pairs.
    """
    all_speakers = list(df['id'].unique())
    speaker_dict = {k: v for k, v in df.groupby('id')}
    final_list = []

    model.return_embedding = True
    model.base_training = False
   
    for speaker in tqdm(all_speakers):
        # Positive pairs
        pos_utt_df = sample_diff_pos_utts(speaker_dict[speaker], n=2*n_candidates)
        pos_utt_pool = pos_utt_df.sample(frac=1).reset_index(drop=True)
        speaker_1 = pos_utt_pool['file_path'].iloc[::2].reset_index(drop=True)
        speaker_2 = pos_utt_pool['file_path'].iloc[1::2].reset_index(drop=True)
        pos_df = pd.DataFrame({
            'speaker_1': speaker_1,
            'speaker_2': speaker_2,
            'y_true': 1
        })

        # Positive utterances for negatives pairs
        pos_utt_df2 = sample_diff_pos_utts(speaker_dict[speaker], n=n_candidates)
        pos_utt_pool2 = pos_utt_df2.sample(frac=1).reset_index(drop=True)

        # Negative pairs: one utterance from current speaker, one from another
        neg_speakers_pool = [s for s in all_speakers if s != speaker]
        neg_speakers_pool = random.sample(neg_speakers_pool, n_candidates)
        neg_samples = pd.concat([speaker_dict[s].sample(1) for s in neg_speakers_pool], ignore_index=True)

        neg_df = pd.DataFrame({
            'speaker_1': pos_utt_pool2['file_path'].reset_index(drop=True),
            'speaker_2': neg_samples['file_path'].reset_index(drop=True),
            'y_true': 0
        })

        # calculate cos_sim for all pairs
        cand_df = pd.concat([pos_df, neg_df], ignore_index=True)
        cand_ds, all_cand_paths = test_dataset(cand_df, batch_size=batch_size)
        y_pred = model.predict(cand_ds, batch_size=batch_size, verbose=0)

        embeddings_dict = {}

        for (path, emb) in zip(all_cand_paths, y_pred):
            embeddings_dict[path] = emb
       
        def get_cosine_similarity(row):
            emb1 = embeddings_dict[row['speaker_1']]
            emb2 = embeddings_dict[row['speaker_2']]
            return np.dot(emb1, emb2)
   
        cand_df['cosine_similarity'] = cand_df.apply(get_cosine_similarity, axis=1)

        # choose hardest n paris
        pos_hard = cand_df[cand_df.y_true == 1].nsmallest(n, 'cosine_similarity')
        neg_hard = cand_df[cand_df.y_true == 0].nlargest(n, 'cosine_similarity')
        # Alternate positive/negative rows
        for positive, negative in zip(pos_hard.to_dict('records'), neg_hard.to_dict('records')):
            final_list.append(positive)
            final_list.append(negative)

    final = pd.DataFrame(final_list)
    final.to_csv(save_path, index=False)
    model.return_embedding = False
    model.base_training = True
    return final