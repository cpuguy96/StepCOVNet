# StepCOVNet

![header_example](https://github.com/cpuguy96/StepCOVNet/blob/master/resources/header_example.gif)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f9f66f23071c45f194cd6b429f2bb508)](https://www.codacy.com/gh/cpuguy96/StepCOVNet/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cpuguy96/StepCOVNet&amp;utm_campaign=Badge_Grade)
[![Pre-submit](https://github.com/cpuguy96/StepCOVNet/actions/workflows/pre-submit.yml/badge.svg)](https://github.com/cpuguy96/StepCOVNet/actions/workflows/pre-submit.yml)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

## Running Audio to SM File Generator

### Currently only produces `.txt` files. Use [`SMDataTools`](https://github.com/jhaco/SMDataTools) to convert `.txt` to `.sm`

```.bash
python stepmania_note_generator.py -i --input <string> -o --output <string> --model <string> -v --verbose <int>
```

* `-i` `--input` input directory path to audio files
* `-o` `--output` output directory path to `.txt` files
* `-m` `--model` input directory path to StepCOVNet model````
* **OPTIONAL:** `-v` `--verbose` `1` shows full verbose, `0` shows no verbose; default is `0`

## Creating Training Dataset

**Link to training data**: [Google Drive](https://drive.google.com/drive/folders/1RNKLXfwTEFdGMCct5bhgIApoNG-8Zg79?usp=drive_link)

To create a training dataset, you need to parse the `.sm` files and convert sound files into `.wav` files:

* [`SMDataTools`](https://github.com/jhaco/SMDataTools) should be used to parse the `.sm` files into `.txt` files.
* [`wav_converter.py`](https://github.com/cpuguy96/StepCOVNet/blob/master/wav_converter.py) can be used to convert the
  audio files into `.wav` files. The default sample rate is `16000hz`.

Once the parsed `.txt` files and `.wav` files are generated, place the `.wav` files into separate directories and
run [`training_data_collection.py`](https://github.com/cpuguy96/StepCOVNet/blob/master/stepcovnet/training_data_collection.py).

```.bash
python training_data_collection.py -w --wav <string> -t --timing <string> -o --output <string> --multi <int> --limit <int> --cores <int> --name <string> --distributed <int>
```

* `-w` `--wav` input directory path to `.wav` files
* `-t` `--timing` input directory path to timing files
* `-o` `--output` output directory path to output dataset
* **OPTIONAL:** `--multi` `1` collects STFTs using `frame_size` of `[2048, 1024, 4096]`, `0` collects STFTs
  using `frame_size` of `[2048]`; default is `0`
* **OPTIONAL:** `--limit` `> 0` stops data collection at limit, `-1` means unlimited; default is `-1`
* **OPTIONAL:** `--cores` `> 0` sets the number of cores to use when collecting data; `-1` means uses the number of
  physical cores; default is `1`
* **OPTIONAL:** `--name` name to give the dataset; default names dataset based on the configuration parameters
* **OPTIONAL:** `--distributed` `0` creates a single dataset, `1` creates a distributed dataset; default is `0`

## Training Model

Once training dataset has been created, run [`train.py`](https://github.com/cpuguy96/StepCOVNet/blob/master/train.py).

```.bash
python train.py -i --input <string> -o --output <string> -d --difficulty <int> --lookback <int> --limit <int> --name <string> --log <string>
``` 

* `-i` `--input` input directory path to training dataset
* `-o` `--output` output directory path to save model
* **OPTIONAL:** `-d` `--difficulty` `[0, 1, 2, 3, 4]` sets the song difficulty to use when training
  to `["challenge", "hard", "medium", "easy", "beginner"]`, respectively; default is `0` or "challenge"
* **OPTIONAL:** `--lookback` `> 2` uses timeseries based on `lookback` when modeling; default is `3`
* **OPTIONAL:** `--limit` `> 0` limits the amount of training samples used during training, `-1` uses all the samples;
  default is `-1`
* **OPTIONAL:** `--name` name to give the finished model; default names model based on dat aset used
* **OPTIONAL:** `--log` output directory path to store tensorboard data

## Credits

* Inspiration from the paper [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
* Most of the source code derived from [musical-onset-efficient](https://github.com/ronggong/musical-onset-efficient)
* [Jhaco](https://github.com/jhaco) for support and collaboration 
