import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input

from stepcovnet.common.tf_config import MIXED_PRECISION_POLICY
from stepcovnet.model.AbstractModel import AbstractModel
from stepcovnet.model.PretrainedModels import PretrainedModels


class ArrowModel(AbstractModel):
    def __init__(self, training_config, architecture=None, name="StepCOVNetArrowModel"):
        arrow_input = Input(shape=training_config.arrow_input_shape, name="arrow_input", dtype=tf.int32)
        arrow_mask = Input(shape=training_config.arrow_mask_shape, name="arrow_mask", dtype=tf.int32)
        model_input = [arrow_input, arrow_mask]

        if architecture is None:
            gp2_model = PretrainedModels.gpt2_model()
            model_output = gp2_model(arrow_input, attention_mask=arrow_mask)[0]
            # GPT-2 model returns feature maps for avg/max pooling. Using LSTM for additional feature extraction.
            # Might be able to replace this with another method in the future
            model_output = GlobalMaxPool1D(dtype=MIXED_PRECISION_POLICY)(model_output)
        else:
            # TODO: Add support for existing arrow models
            raise NotImplementedError("No support yet for existing architectures")

        super(ArrowModel, self).__init__(model_input=model_input, model_output=model_output, name=name)
