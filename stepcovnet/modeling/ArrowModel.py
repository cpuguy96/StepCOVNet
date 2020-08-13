import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM

from stepcovnet.modeling.AbstractModel import AbstractModel
from stepcovnet.modeling.PretrainedModels import PretrainedModels


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
            model_output = Bidirectional(LSTM(64, return_sequences=True, kernel_initializer=glorot_uniform(42))
                                         )(model_output)
            model_output = GlobalMaxPool1D()(model_output)
        else:
            # TODO: Add support for existing arrow models
            raise NotImplementedError("No support yet for existing architectures")

        super(ArrowModel, self).__init__(model_input=model_input, model_output=model_output, name=name)
