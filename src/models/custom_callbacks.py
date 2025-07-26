import tensorflow as tf
import pandas as pd
from hydra.utils import to_absolute_path
from src.eval import eer

class oldEERMonitor(tf.keras.callbacks.Callback):
    """
    Custom Keras callback for monitoring Equal Error Rate (EER) during training.

    Features:
    - Computes EER at the end of each epoch using the provided `eer_fn` function.
    - Saves the model whenever EER improves.
    - If the improvement in EER is less than `improvement_threshold`, reduces the learning rate by `factor`.
    - If no improvement occurs for `patience` consecutive epochs, also reduces the learning rate.

    Usage example:
    --------------
    def eer(model):
        # Your custom function that calculates EER for the given model.
        # Should return a float value (lower = better).
        ...

    eer_callback = EERMonitor(     
        model_path="models/best_model.keras",   # Where to save the best model
        eer_fn=eer,                             # Your EER evaluation function
        patience=1,                             # Number of epochs with no improvement before reducing LR
        factor=0.1,                             # Factor to reduce LR by
        min_lr=1e-8,                            # Minimum LR
        improvement_threshold=0.05              # Minimum improvement considered significant
    )
    """
    def __init__(
        self,
        model_path: str,
        eer_fn: callable = eer,
        patience=0,
        factor=0.5,
        min_lr=1e-8,
        improvement_threshold=0.03
    ):
        super().__init__()
        self.eer_fn = eer_fn
        self.model_path = model_path
        self.best_eer = float('inf')
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.bad_epochs = 0
        self.improvement_threshold = improvement_threshold

    def on_epoch_end(self, epoch, logs=None):
        current_eer = self.eer_fn(self.model)
        print(f"\nEpoch {epoch+1} — EER: {current_eer:.4f} (Best: {self.best_eer:.4f})")
        improvement = self.best_eer - current_eer
       
        if current_eer < self.best_eer:
            print("Improved EER — saving model.")
            self.model.save(self.model_path)
            # if improvement >=  improvement_threshold [%] — reduce LR
            if improvement < self.improvement_threshold:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"Improvement was too small ({improvement:.4f})! Reducing learning rate: {old_lr:.2e} → {new_lr:.2e}")
            self.best_eer = current_eer
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            print(f"No improvement in EER. Patience: {self.bad_epochs}/{self.patience}")
            if self.bad_epochs > self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"Reducing learning rate: {old_lr:.2e} → {new_lr:.2e}")
                self.bad_epochs = 0

class EERMonitor(tf.keras.callbacks.Callback):
    """
    Keras callback for monitoring Equal Error Rate (EER) during training.

    Features:
        - Computes and logs EER to TensorBoard at specified batches (`steps_interval`).
        - Saves the model only when EER improves.

    Args:
        eer_fn (callable): Function with signature `eer_fn(model, test_df)` returning EER (float).
        model_path (str): Path where the best model will be saved.
        test_df: DataFrame for EER evaluation.
        patience (int): Number of epochs with no improvement before stopping training.
        log_dir (str, optional): Directory for TensorBoard logs.
        steps_interval (list[int], optional): List of batch numbers (starting from 1) on which EER should be evaluated and logged.

    Example:
        eer_callback = EERMonitor( 
            model_path="models/best_model.keras",
            test_df=val_df,
            eer_fn=eer,
            patience=3,
            log_dir="logs/eer",
            steps_interval=[500, 1000, 2000]
        )
        model.fit(..., callbacks=[eer_callback])
    """
    def __init__(self,
                 model_path,
                 test_df_path: str,
                 eer_fn: callable = eer,
                 log_dir=None,
                 patience=0,
                 steps_interval=None):
        super().__init__()
        self.model_path = model_path
        self.test_df =  pd.read_csv(to_absolute_path(test_df_path))
        self.eer_fn = eer_fn
        self.best_eer = float('inf')
        self.patience = patience
        self.steps_interval = steps_interval if steps_interval else []
        self.writer = tf.summary.create_file_writer(log_dir) if log_dir else None
        self.batch_counter = 0
        self.bad_epochs = 0

    def _handle_eer(self, current_eer, context):
        print(f"\n[{context}] EER: {current_eer:.4f} (Best: {self.best_eer:.4f})")
        if current_eer < self.best_eer:
            print(f"EER improved by {self.best_eer - current_eer:.4f} — saving model.")
            self.model.save(self.model_path)
            self.best_eer = current_eer
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            print(f"No improvement in EER. Patience: {self.bad_epochs}/{self.patience}")
            if self.bad_epochs >= self.patience:
                self.bad_epochs = 0
                self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.steps_interval and (self.batch_counter in self.steps_interval):
            current_eer = self.eer_fn(self.model, self.test_df)
            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.scalar("EER", current_eer, step=self.batch_counter)
                    self.writer.flush()
            self._handle_eer(current_eer, f"batch {self.batch_counter}")

    def on_epoch_end(self, epoch, logs=None):
        current_eer = self.eer_fn(self.model, self.test_df)
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar("EER", current_eer, step=self.batch_counter)
                self.writer.flush()
        self._handle_eer(current_eer, f"epoch {epoch+1}")

