import tensorflow as tf


class TrainingHyperparameters(object):
    DEFAULT_METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='pre'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.AUC(curve="PR", name='pr_auc'),
        tf.keras.metrics.AUC(name='auc')
    ]
    DEFAULT_LOSS = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    DEFAULT_OPTIMIZER = tf.keras.optimizers.Nadam(beta_1=0.99)
    DEFAULT_EPOCHS = 2
    DEFAULT_PATIENCE = 5
    DEFAULT_BATCH_SIZE = 2

    def __init__(self, optimizer=None, loss=None, metrics=None, batch_size=None, epochs=None, patience=None,
                 log_path=None, retrain=None):
        self.optimizer = optimizer if optimizer is not None else self.DEFAULT_OPTIMIZER
        self.loss = loss if loss is not None else self.DEFAULT_LOSS
        self.metrics = metrics if metrics is not None else self.DEFAULT_METRICS
        self.patience = patience if patience is not None else self.DEFAULT_PATIENCE
        self.epochs = epochs if epochs is not None else self.DEFAULT_EPOCHS
        self.batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        self.retrain = retrain if retrain is not None else True
        self.log_path = log_path
