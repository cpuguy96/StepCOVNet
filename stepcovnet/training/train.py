import datetime
import os

from stepcovnet.configuration.parameters import NUM_FREQ_BANDS, NUM_MULTI_CHANNELS, NUM_TIME_BANDS, BATCH_SIZE
from stepcovnet.training.modeling import prepare_model


def train(input_path, output_path, multi_int, extra_int, under_sample_int, lookback, limit, name, log_path,
          pretrained_model_path, model_type_int):
    if not os.path.isdir(input_path):
        raise NotADirectoryError('Input path %s not found' % input_path)

    if not os.path.isdir(output_path):
        print("Model output path not found. Creating directory...")
        os.makedirs(output_path, exist_ok=True)

    if lookback <= 0:
        raise ValueError('Lookback needs to be >= 1')

    if lookback > BATCH_SIZE:
        raise ValueError('Lookback needs to be <= BATCH_SIZE (currently %s)' % BATCH_SIZE)

    if lookback > 1 and under_sample_int == 1:
        raise ValueError('Cannot use under sample when lookback > 1')

    if limit == 0:
        raise ValueError('Limit cannot be = 0')

    if name is not None and not name:
        raise ValueError('Model name cannot be empty')

    if pretrained_model_path is not None and not os.path.isfile(pretrained_model_path):
        raise FileNotFoundError('Pretrained model path %s not found' % pretrained_model_path)

    if log_path is not None and not os.path.isdir(log_path):
        print("Log output path not found. Creating directory...")
        os.makedirs(log_path, exist_ok=True)

    if log_path is not None:
        log_path = os.path.join(log_path, "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    multi = True if multi_int == 1 else False
    extra = True if extra_int == 1 else False
    under_sample = True if under_sample_int == 1 else False

    model_type = "normal" if model_type_int == 0 else "paper"

    built_model_name = "multi_" if multi else ""
    built_model_name += "under_" if under_sample else ""

    filename_labels = os.path.join(input_path, built_model_name + 'labels.npz')
    filename_sample_weights = os.path.join(input_path, built_model_name + 'sample_weights.npz')

    if extra:
        path_extra_features = os.path.join(input_path, built_model_name + 'extra_features.npz')
        extra_input_shape = (None, 2)
    else:
        path_extra_features = None
        extra_input_shape = None

    filename_scaler = []
    if multi:
        input_shape = (lookback, NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS) if lookback > 1 else (
            NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS)
        filename_scaler.append(os.path.join(input_path, built_model_name + 'scaler_low.pkl'))
        filename_scaler.append(os.path.join(input_path, built_model_name + 'scaler_mid.pkl'))
        filename_scaler.append(os.path.join(input_path, built_model_name + 'scaler_high.pkl'))
    else:
        input_shape = (lookback, 1, NUM_FREQ_BANDS, NUM_TIME_BANDS) if lookback > 1 else (
            1, NUM_FREQ_BANDS, NUM_TIME_BANDS)
        filename_scaler.append(os.path.join(input_path, built_model_name + 'scaler.pkl'))
    filename_features = os.path.join(input_path, built_model_name + 'dataset_features.npz')

    # adding this afterwards since we want to only rename the model
    built_model_name += "extra_" if extra else ""
    built_model_name += "time%s_" % lookback if lookback > 1 else ""
    built_model_name += "pretrained_" if pretrained_model_path is not None else ""
    # finally, specify training timing model
    built_model_name += "timing_model"

    model_name = name if name is not None else built_model_name
    log_path = None if log_path is None else log_path + "_" + model_name

    file_path_model = os.path.join(output_path)

    prepare_model(filename_features, filename_labels, filename_sample_weights, filename_scaler, input_shape=input_shape,
                  model_name=model_name, model_out_path=file_path_model, extra_input_shape=extra_input_shape,
                  path_extra_features=path_extra_features, lookback=lookback, multi=multi, limit=limit,
                  log_path=log_path, filename_pretrained_model=pretrained_model_path, model_type=model_type)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train a note timings model")

    parser.add_argument("-i", "--input",
                        type=str,
                        help="Input training data path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output stored model path")
    parser.add_argument("--multi",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Whether multiple STFT window time-lengths are used in training: 0 - single, 1 - multi")
    parser.add_argument("--extra",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Whether to use extra data from madmom and librosa: 0 - not used, 1 - used")
    parser.add_argument("--under_sample",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Whether to under sample for balanced classes: 0 - not used, 1 - used")
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
    parser.add_argument("--pretrained_model",
                        type=str,
                        default=None,
                        help="Input path to pretrained model to use transfer learning")
    parser.add_argument("--model_type",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Type of model architecture to train with: 0 use custom model, 1 use model configuration "
                             "given in DDC paper")
    parser.add_argument("--log",
                        type=str,
                        default=None,
                        help="Output log data path")
    args = parser.parse_args()

    train(args.input, args.output, args.multi, args.extra, args.under_sample, args.lookback, args.limit, args.name,
          args.log, args.pretrained_model, args.model_type)
