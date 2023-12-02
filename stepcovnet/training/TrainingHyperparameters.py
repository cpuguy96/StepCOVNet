"""Configurable training hyperparameters"""
# TODO(https://github.com/cpuguy96/StepCOVNet/issues/2):
#  Move all training hyperparameters into config file
from keras import metrics, optimizers, losses


class TrainingHyperparameters(object):
    DEFAULT_METRICS = [
        metrics.CategoricalAccuracy(name="acc"),
        metrics.Precision(name="pre"),
        metrics.Recall(name="rec"),
        metrics.AUC(curve="PR", name="pr_auc"),
        metrics.AUC(name="auc"),
    ]
    DEFAULT_LOSS = losses.CategoricalCrossentropy(label_smoothing=0.05)
    DEFAULT_OPTIMIZER = optimizers.Nadam(beta_1=0.99)
    DEFAULT_EPOCHS = 15
    DEFAULT_PATIENCE = 3
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        optimizer=None,
        loss=None,
        hp_metrics=None,
        batch_size=None,
        epochs=None,
        patience=None,
        log_path=None,
        retrain=None,
    ):
        self.optimizer = optimizer if optimizer is not None else self.DEFAULT_OPTIMIZER
        self.loss = loss if loss is not None else self.DEFAULT_LOSS
        self.metrics = hp_metrics if hp_metrics is not None else self.DEFAULT_METRICS
        self.patience = patience if patience is not None else self.DEFAULT_PATIENCE
        self.epochs = epochs if epochs is not None else self.DEFAULT_EPOCHS
        self.batch_size = (
            batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        )
        self.retrain = retrain if retrain is not None else True
        self.log_path = log_path

    def __str__(self):
        str_dict = {}
        for key, value in self.__dict__.items():
            str_dict[key] = str(value)
        return str_dict.__str__()
