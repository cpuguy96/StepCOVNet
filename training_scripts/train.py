import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To train the model.")

    parser.add_argument("--path_input",
                        type=str,
                        help="Input path where you put the training data")
    parser.add_argument("--path_output",
                        type=str,
                        help="Output path where you store the models and logs")
    parser.add_argument("--pretrained",
                        type=str,
                        help="Input path where a pretrained model is located")
    parser.add_argument("--multi",
                        type=int,
                        default=0,
                        help="whether multiple STFT window time-lengths are used in training ")
    parser.add_argument("--under_sample",
                        type=int,
                        default=0,
                        help="whether to under sample for balanced classes")
    args = parser.parse_args()

    if not os.path.isdir(args.path_input):
        raise OSError('Input path %s not found' % args.path_input)

    if not os.path.isdir(args.path_output):
        raise OSError('Output path %s not found' % args.path_output)

    if len(args.pretrained) > 0 and not os.path.isfile(args.pretrained):
        raise OSError('Pretrained model %s is not found' % args.pretrained)

    filename_scaler = []

    if args.multi == 1:
        multi = True
    else:
        multi = False

    if args.under_sample == 1:
        under_sample = True
    else:
        under_sample = False

    prefix = ""

    if multi:
        prefix += "multi_"

    if under_sample:
        prefix += "under_"

    filename_labels_train_validation_set = os.path.join(args.path_input, prefix + 'labels.npz')
    filename_sample_weights = os.path.join(args.path_input, prefix + 'sample_weights.npz')

    if multi:
        input_shape = (80, 15, 3)
        channel = 3

        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler_low.pkl'))
        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler_mid.pkl'))
        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler_high.pkl'))

        filename_train_validation_set = os.path.join(args.path_input, prefix + 'multi_dataset_features.npz')
    else:
        input_shape = (80, 15)
        channel = 1

        filename_scaler.append(os.path.join(args.path_input, prefix + 'scaler.pkl'))

        filename_train_validation_set = os.path.join(args.path_input, prefix + 'dataset_features.npz')

    if args.pretrained is not None:
        filename_pretrained_model = os.path.join(args.pretrained)
        file_path_model = os.path.join(args.path_output, 'pretrained_trained_model.h5')
        file_path_log = os.path.join(args.path_output, 'pretrained_trained_model.csv')
        train_model(filename_train_validation_set,
                    filename_labels_train_validation_set,
                    filename_sample_weights,
                    filename_scaler,
                    input_shape=input_shape,
                    file_path_model=file_path_model,
                    filename_log=file_path_log,
                    channel=channel,
                    pretrained_model=filename_pretrained_model)
    else:
        file_path_model = os.path.join(args.path_output, 'trained_model.h5')
        file_path_log = os.path.join(args.path_output, 'trained_model.csv')
        train_model(filename_train_validation_set,
                    filename_labels_train_validation_set,
                    filename_sample_weights,
                    filename_scaler,
                    input_shape=input_shape,
                    file_path_model=file_path_model,
                    filename_log=file_path_log,
                    channel=channel)
