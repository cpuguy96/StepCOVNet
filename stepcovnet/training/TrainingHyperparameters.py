import tensorflow as tf

from stepcovnet.data.Tokenizers import Tokenizers


class TrainingHyperparameters(object):
    DEFAULT_METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='pre'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.AUC(curve="PR", name='pr_auc'),
        tf.keras.metrics.AUC(name='auc')
    ]
    DEFAULT_LOSS = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    DEFAULT_OPTIMIZER = tf.keras.optimizers.Nadam(beta_1=0.99)
    DEFAULT_EPOCHS = 15
    DEFAULT_PATIENCE = 3
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_TOKENIZER = Tokenizers.GPT2.name

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
