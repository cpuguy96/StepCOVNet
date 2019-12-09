from scripts_training.modeling import train_model

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="To train the model.")

    parser.add_argument("--path_input",
                        type=str,
                        help="Input path where you put the training data")
    parser.add_argument("--path_output",
                        type=str,
                        help="Output path where you store the models and logs")
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
    args = parser.parse_args()

    if not os.path.isdir(args.path_input):
        raise NotADirectoryError('Input path %s not found' % args.path_input)

    if not os.path.isdir(args.path_output):
        print("Model output path not found. Creating directory...")
        os.makedirs(args.path_output, exist_ok=True)

    if args.lookback <= 0:
        raise ValueError('Lookback needs to be >= 1')

    if args.limit == 0:
        raise ValueError('Limit cannot be = 0')

    if args.lookback > 1 and args.under_sample == 1:
        raise ValueError('Cannot use under sample when lookback > 1')

    filename_scaler = []

    if args.multi == 1:
        multi = True
    else:
        multi = False

    if args.extra == 1:
        extra = True
    else:
        extra = False

    if args.under_sample == 1:
        under_sample = True
    else:
        under_sample = False

    lookback = args.lookback

    prefix = ""

    if multi:
        prefix += "multi_"

    if under_sample:
        prefix += "under_"

    filename_labels = os.path.join(args.path_input, prefix + 'labels.npz')
    filename_sample_weights = os.path.join(args.path_input, prefix + 'sample_weights.npz')

    if extra:
        path_extra_features = os.path.join(args.path_input, prefix + 'extra_features.npz')
        extra_input_shape = (None, 2)
    else:
        path_extra_features = None
        extra_input_shape = None

    if multi:
        input_shape = (80, 15, 3)
        if lookback > 1:
            input_shape = (lookback, 80, 15, 3)

        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler_low.pkl'))
        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler_mid.pkl'))
        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler_high.pkl'))
    else:
        input_shape = (1, 80, 15)
        if lookback > 1:
            input_shape = (lookback, 1, 80, 15)

        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler.pkl'))
    filename_features = os.path.join(args.path_input, prefix + 'dataset_features.npz')

    # adding this afterwards since we want to only rename the model
    if extra:
        prefix += "extra_"

    # adding this afterwards since don't have preprocessing for timeseries data set up
    # might add this preprocessing later if training seems to be slow to start
    if lookback > 1:
        prefix += "time_"

    file_path_model = os.path.join(args.path_output)

    train_model(filename_features,
                filename_labels,
                filename_sample_weights,
                filename_scaler,
                input_shape=input_shape,
                prefix=prefix,
                model_out_path=file_path_model,
                extra_input_shape=extra_input_shape,
                path_extra_features=path_extra_features,
                lookback=lookback,
                limit=args.limit)


if __name__ == '__main__':
    main()
