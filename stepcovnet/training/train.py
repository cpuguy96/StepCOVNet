import datetime
import json
import os
import pickle

from stepcovnet.dataset.ModelDatasetTypes import ModelDatasetTypes
from stepcovnet.modeling.ArrowModel import ArrowModel
from stepcovnet.modeling.AudioModel import AudioModel
from stepcovnet.modeling.ClassifierModel import ClassifierModel
from stepcovnet.modeling.StepCOVNetModel import StepCOVNetModel
from stepcovnet.training.TrainingConfig import TrainingConfig
from stepcovnet.training.TrainingExecutor import TrainingExecutor
from stepcovnet.training.TrainingHyperparameters import TrainingHyperparameters
from stepcovnet.training.TrainingInput import TrainingInput


def load_training_data(input_path):
    metadata = json.load(open(os.path.join(input_path, "metadata.json"), 'r'))
    dataset_name = metadata["dataset_name"]
    dataset_type = metadata["dataset_type"]
    dataset = ModelDatasetTypes[dataset_type].value(os.path.join(input_path, dataset_name + "_dataset.hdf5"))
    scalers = pickle.load(open(os.path.join(input_path, dataset_name + "_scaler.pkl"), 'rb'))
    dataset_config = metadata["config"]
    return dataset, scalers, dataset_config


def run_training(input_path, output_path, model_name, limit, lookback, difficulty, log_path):
    dataset, scalers, dataset_config = load_training_data(input_path)

    hyperparameters = TrainingHyperparameters(log_path=log_path)

    training_config = TrainingConfig(dataset=dataset, dataset_config=dataset_config, hyperparameters=hyperparameters,
                                     all_scalers=scalers, limit=limit, lookback=lookback, difficulty=difficulty)
    training_input = TrainingInput(dataset, training_config)

    arrow_model = ArrowModel(training_input.training_config)
    audio_model = AudioModel(training_input.training_config)
    model = ClassifierModel(training_input.training_config, arrow_model, audio_model)
    stepcovnet_model = StepCOVNetModel(model_path=output_path, model_name=model_name, model=model)

    TrainingExecutor(training_input=training_input, stepcovnet_model=stepcovnet_model).execute()


def train(input_path, output_path, difficulty_int, lookback, limit, name, log_path):
    if not os.path.isdir(input_path):
        raise NotADirectoryError('Input path %s not found' % input_path)

    if not os.path.isdir(output_path):
        print("Model output path not found. Creating directory...")
        os.makedirs(output_path, exist_ok=True)

    if lookback <= 0:
        raise ValueError('Lookback needs to be >= 1')

    if limit == 0:
        raise ValueError('Limit cannot be = 0')

    if name is not None and not name:
        raise ValueError('Model name cannot be empty')

    if log_path is not None and not os.path.isdir(log_path):
        print("Log output path not found. Creating directory...")
        os.makedirs(log_path, exist_ok=True)

    if log_path is not None:
        log_path = os.path.join(log_path, "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    difficulty = ["challenge", "hard", "medium", "easy", "beginner"][difficulty_int]

    built_model_name = "time%s_" % lookback if lookback > 1 else ""
    built_model_name += "%s_" % difficulty
    # finally, specify training timing model
    built_model_name += "timing_model"

    model_name = name if name is not None else built_model_name
    log_path = None if log_path is None else log_path + "_" + model_name

    output_path = os.path.join(output_path, model_name)
    os.makedirs(output_path, exist_ok=True)

    run_training(input_path=input_path, output_path=output_path, model_name=model_name, limit=limit, lookback=lookback,
                 difficulty=difficulty, log_path=log_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train a note timings model")

    parser.add_argument("-i", "--input",
                        type=str,
                        help="Input training data path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output stored model path")
    parser.add_argument("-d", "--difficulty",
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4],
                        help="Game difficulty to use when training: 0 - challenge, 1 - hard, 2 - medium, 3 - easy, 4, "
                             "- beginner")
    parser.add_argument("--lookback",
                        type=int,
                        default=1,
                        help="Number of frames to lookback when training: 1 - non timeseries, > 1 timeseries")
    parser.add_argument("--limit",
                        type=int,
                        default=-1,
                        help="Maximum number of frames to use when training: -1 unlimited, > 0 frame limit")
    parser.add_argument("--name",
                        type=str,
                        default=None,
                        help="Name to give finished model")
    parser.add_argument("--log",
                        type=str,
                        default=None,
                        help="Output log data path for tensorboard")
    args = parser.parse_args()

    train(input_path=args.input, output_path=args.output, difficulty_int=args.difficulty,
          lookback=args.lookback, limit=args.limit, name=args.name, log_path=args.log)
