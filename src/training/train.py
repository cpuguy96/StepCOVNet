import os

from configuration.parameters import NUM_FREQ_BANDS, NUM_MULTI_CHANNELS, NUM_TIME_BANDS
from training.modeling import prepare_model


def train(input_path,
          output_path,
          multi_int,
          extra_int,
          under_sample_int,
          lookback,
          limit,
          name,
          pretrained_model_path):
    if not os.path.isdir(input_path):
        raise NotADirectoryError('Input path %s not found' % input_path)

    if not os.path.isdir(output_path):
        print("Model output path not found. Creating directory...")
        os.makedirs(output_path, exist_ok=True)

    if lookback <= 0:
        raise ValueError('Lookback needs to be >= 1')

    if limit == 0:
        raise ValueError('Limit cannot be = 0')

    if lookback > 1 and under_sample_int == 1:
        raise ValueError('Cannot use under sample when lookback > 1')

    if name is not None and not name:
        raise ValueError('Model name cannot be empty')

    if pretrained_model_path is not None and not os.path.isfile(pretrained_model_path):
        raise FileNotFoundError('Pretrained model path %s not found' % pretrained_model_path)

    multi = True if multi_int == 1 else False
    extra = True if extra_int == 1 else False
    under_sample = True if under_sample_int == 1 else False

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

    file_path_model = os.path.join(output_path)

    prepare_model(filename_features, filename_labels, filename_sample_weights, filename_scaler, input_shape=input_shape,
                  model_name=model_name, model_out_path=file_path_model,
                  extra_input_shape=extra_input_shape, path_extra_features=path_extra_features, lookback=lookback,
                  multi=multi,
                  limit=limit, filename_pretrained_model=pretrained_model_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="To train the model.")

    parser.add_argument("-i", "--input",
                        type=str,
                        help="Input path where you put the training data")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output path where you store the model")
    parser.add_argument("--multi",
                        type=int,
                        default=0,
                        help="whether multiple STFT window time-lengths are used in training ")
    parser.add_argument("--extra",
                        type=int,
                        default=0,
                        help="whether to use extra data from madmom and librosa")
    parser.add_argument("--under_sample",
                        type=int,
                        default=0,
                        help="whether to under sample for balanced classes")
    parser.add_argument("--lookback",
                        type=int,
                        default=1,
                        help="number of frames to lookback when training")
    parser.add_argument("--limit",
                        type=int,
                        default=-1,
                        help="number of frames to use when training")
    parser.add_argument("--name",
                        type=str,
                        default=None,
                        help="Name to give finished model. Defaults to dataset description.")
    parser.add_argument("--pretrained_model",
                        type=str,
                        default=None,
                        help="Input path to pretrained model to use transfer learning.")
    args = parser.parse_args()

    train(args.input,
          args.output,
          args.multi,
          args.extra,
          args.under_sample,
          args.lookback,
          args.limit,
          args.name,
          args.pretrained_model)
