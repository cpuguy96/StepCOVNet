import tensorflow as tf
from transformers import GPT2Tokenizer


class TrainingHyperparameters(object):
    DEFAULT_METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='pre'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.AUC(curve="PR", name='pr_auc'),
        tf.keras.metrics.AUC(name='auc')
    ]
    DEFAULT_LOSS = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    DEFAULT_OPTIMIZER = tf.keras.optimizers.Nadam(beta_1=0.99)
    DEFAULT_EPOCHS = 10
    DEFAULT_PATIENCE = 2
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')

    def __init__(self, optimizer=None, loss=None, metrics=None, batch_size=None, epochs=None, patience=None,
                 log_path=None, retrain=None, tokenizer=None):
        self.optimizer = optimizer if optimizer is not None else self.DEFAULT_OPTIMIZER
        self.loss = loss if loss is not None else self.DEFAULT_LOSS
        self.metrics = metrics if metrics is not None else self.DEFAULT_METRICS
        self.patience = patience if patience is not None else self.DEFAULT_PATIENCE
        self.epochs = epochs if epochs is not None else self.DEFAULT_EPOCHS
        self.batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        self.retrain = retrain if retrain is not None else True
        self.log_path = log_path
        self.tokenizer = tokenizer if tokenizer is not None else self.DEFAULT_TOKENIZER

    def __str__(self):
        str_dict = {}
        for key, value in self.__dict__.items():
            str_dict[key] = str(value)
        return str_dict.__str__()

    @staticmethod
    def macro_double_soft_f1(y, y_hat):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
        soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
        cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
        macro_cost = tf.reduce_mean(cost)  # average on all labels
        return macro_cost
