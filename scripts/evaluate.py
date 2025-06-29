import argparse
import pandas as pd
import tensorflow as tf
from src.models import L2Normalization, CosineLayer, AdaCosLoss, AdaCosLossMargin, LMCLoss, VerificationModel
from src.eval import validate_on_testset
from src.utils import plot_eer_curve
from classification_models.tfkeras import Classifiers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a test set and compute EER.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .keras model file.")
    parser.add_argument("--test_csv", type=str, required=True, help="CSV with columns: speaker_1, speaker_2, y_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default=None, help="Where to save EER results (optional)")
    parser.add_argument("--plot", action="store_true", help="Display EER curve after evaluation")
    args = parser.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path)
    
    print("Loading test data...")
    test_df = pd.read_csv(args.test_csv)

    print("Running evaluation...")
    eer_df, test_df_out, eer_row = validate_on_testset(model, test_df, batch_size=args.batch_size)

    #print("Best EER threshold row:\n", eer_row)
    if args.output:
        eer_df.to_csv(args.output, index=False)
        print(f"EER results saved to {args.output}")

    if args.plot:
        plot_eer_curve(eer_df)
