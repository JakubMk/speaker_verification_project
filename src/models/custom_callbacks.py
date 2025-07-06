import tensorflow as tf
from src.eval import eer

class EERMonitor(tf.keras.callbacks.Callback):
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
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(f"Improvement was too small ({improvement:.4f})! Reducing learning rate: {old_lr:.2e} → {new_lr:.2e}")
            self.best_eer = current_eer
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            print(f"No improvement in EER. Patience: {self.bad_epochs}/{self.patience}")
            if self.bad_epochs > self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = max(old_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(f"Reducing learning rate: {old_lr:.2e} → {new_lr:.2e}")
                self.bad_epochs = 0