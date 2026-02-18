# StepCOVNet

![header_example](resources/header_example.gif)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f9f66f23071c45f194cd6b429f2bb508)](https://www.codacy.com/gh/cpuguy96/StepCOVNet/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cpuguy96/StepCOVNet&amp;utm_campaign=Badge_Grade)
[![Pre-submit](https://github.com/cpuguy96/StepCOVNet/actions/workflows/pre-submit.yml/badge.svg)](https://github.com/cpuguy96/StepCOVNet/actions/workflows/pre-submit.yml)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

## Running Audio to SM File Generator

### Currently only produces `.txt` files. Use [`SMDataTools`](https://github.com/jhaco/SMDataTools) to convert `.txt` to `.sm`

```.bash
python scripts/generate.py --audio_path <string> --song_title <string> --bpm <int> --onset_model_path <string> --arrow_model_path <string> --output_file <string>
```

* `--audio_path` input audio file path
* `--song_title` title of the song
* `--bpm` BPM of the song
* `--onset_model_path` path to the trained onset detection model (.keras)
* `--arrow_model_path` path to the trained arrow prediction model (.keras)
* `--output_file` output file path for the generated chart

## Creating Training Dataset

**Link to training data**:
[Google Drive](https://drive.google.com/drive/folders/1Etkj3f-lHM2Y2eH-9el_emUwZXif2GKD?usp=sharing)

To create a training dataset, you need to parse the `.sm` files into `.txt` files:

* [`SMDataTools`](https://github.com/jhaco/SMDataTools) should be used to parse the `.sm` files into `.txt` files.

The training scripts expect a directory structure where audio files and their corresponding `.txt` chart files (with the
same filename stem) are located together.
Supported audio formats are `.mp3`, `.ogg`, and `.wav`.

## Training Model

Once training dataset has been created, run `scripts/train_onset.py` or `scripts/train_arrow.py`.

### Training Onset Model

```.bash
python scripts/train_onset.py --train_data_dir <string> --val_data_dir <string> --model_output_dir <string> --epochs <int> --callback_root_dir <string> --take_count <int> --model_name <string>
```

* `--train_data_dir` directory containing training data
* `--val_data_dir` directory containing validation data
* `--model_output_dir` directory where the trained model will be saved
* **OPTIONAL:** `--epochs` number of epochs to train for; default is `10`
* **OPTIONAL:** `--callback_root_dir` root directory for storing training callbacks (checkpoints, logs)
* **OPTIONAL:** `--take_count` number of batches to use from the training dataset; default is `1`
* **OPTIONAL:** `--model_name` name of the model

### Training Arrow Model

```.bash
python scripts/train_arrow.py --train_data_dir <string> --val_data_dir <string> --model_output_dir <string> --epochs <int> --callback_root_dir <string> --take_count <int> --model_name <string>
```

* `--train_data_dir` directory containing training data
* `--val_data_dir` directory containing validation data
* `--model_output_dir` directory where the trained model will be saved
* **OPTIONAL:** `--epochs` number of epochs to train for; default is `10`
* **OPTIONAL:** `--callback_root_dir` root directory for storing training callbacks (checkpoints, logs)
* **OPTIONAL:** `--take_count` number of batches to use from the training dataset; default is `1`
* **OPTIONAL:** `--model_name` name of the model

## Credits

* Inspiration from the paper [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
* Most of the source code derived from [musical-onset-efficient](https://github.com/ronggong/musical-onset-efficient)
* [Jhaco](https://github.com/jhaco) for support and collaboration
