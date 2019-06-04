import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './jingju_crnn')))

from models import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To train the model.")

    parser.add_argument("--path_input",
                        type=str,
                        help="Input path where you put the training data")

    parser.add_argument("--path_output",
                        type=str,
                        help="Output path where you store the models and logs")

    args = parser.parse_args()

    filename_train_validation_set = os.path.join(args.path_input, 'dataset_features.h5')
    filename_labels_train_validation_set = os.path.join(args.path_input, 'dataset_labels.pkl')
    filename_sample_weights = os.path.join(args.path_input, 'dataset_sample_weights.pkl')

    file_path_model = os.path.join(args.path_output, 'model.h5')
    file_path_log = os.path.join(args.path_output, 'model.csv')
    train_model(filename_train_validation_set,
                filename_labels_train_validation_set,
                filename_sample_weights,
                filter_density=1,
                dropout=0.5,
                input_shape=(80, 15),
                file_path_model=file_path_model,
                filename_log=file_path_log,
                channel=1)
