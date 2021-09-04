from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer

from stepcovnet.model.ArrowModel import ArrowModel
from stepcovnet.model.PretrainedModels import PretrainedModels


class GPT2ArrowModel(ArrowModel):
    def _create_arrow_model(self, arrow_input: Input, arrow_mask: Input) -> Layer:
        gp2_model = PretrainedModels.gpt2_model()
        model_output = gp2_model(arrow_input, attention_mask=arrow_mask)[0]
        # GPT-2 model returns feature maps for avg/max pooling. Using LSTM for additional feature extraction.
        # Might be able to replace this with another method in the future
        return GlobalMaxPool1D()(model_output)
