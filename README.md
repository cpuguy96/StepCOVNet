# StepCOVNet
![header_example](https://github.com/cpuguy96/StepCOVNet/blob/master/resources/header_example.gif)

[![Coverage Status](https://coveralls.io/repos/github/cpuguy96/StepCOVNet/badge.svg?branch=master)](https://coveralls.io/github/cpuguy96/StepCOVNet?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a4ef846886e446229d04974cde24c6dd)](https://www.codacy.com/manual/cpuguy96/StepCOVNet?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cpuguy96/StepCOVNet&amp;utm_campaign=Badge_Grade)
![](https://github.com/cpuguy96/StepCOVNet/workflows/StepCOVNet%20Application/badge.svg)

## Running Audio to SM File Generator
### Currently only produces `.txt` files. Use [`smfile_writer.py`](https://github.com/jhaco/SMFile_Writer) to convert `.txt` to `.sm`
```.bash
python stepmania_note_generator.py -i --input <string> -o --output <string> -s --scalers <string> --timing_model <string> --arrow_model <string> -v --verbose <int>
```
* `-i` `--input` input directory path to audio files
* `-o` `--output` output directory path to `.sm` files
* **OPTIONAL:** `-s` `--scalers` input directory path to scalers used in training; default is `"training_data/"`
* **OPTIONAL:** `--timing_model` input directory path to beat timing model; default is `"models/timing_model.h5"`
* **OPTIONAL:** `--arrow_model` input directory path to arrow selection model; default is `"models/retrained_arrow_model.h5"`
* **OPTIONAL:** `-v` `--verbose` `1` shows full verbose, `0` shows no verbose; default is `0`




## Creating Dataset
To create a training dataset, you need to parse the `.sm` files and convert sound files into `.wav` files: 
* [`smfile_parser.py`](https://github.com/jhaco/SMFile_Parser) should be used to parse the `.sm` files into `.txt` files. 
* [`wav_converter.py`](https://github.com/cpuguy96/StepCOVNet/blob/master/stepcovnet/wrapper/wav_converter.py) can be used to convert the audio files into `.wav` files. If a different `.wav` converter is uses, ensure the sample rate is `44100hz`.

Once the parsed `.txt` files and `.wav` files are generated, place the `.wav` files into separate directories and run [`training_data_collection.py`](https://github.com/cpuguy96/StepCOVNet/blob/master/stepcovnet/data_collection/training_data_collection.py).

```.bash
cd stepcovnet/data_collection
python training_data_collection.py -w --wav <string> -t --timing <string> -o --output <string> --multi <int> --extra <int> --under_sample <int> --limit <int>
```
* `-w` `--wav` input directory path to `.wav` files
* `-t` `--timing` input directory path to timing files
* `-o` `--output` output directory path to output dataset
*  **OPTIONAL:** `--multi` `1` collects STFTs using `frame_size` of `[2048, 1024, 4096]`, `0` collects STFTs using `frame_size` of `[2048]`; default is `0`
* **OPTIONAL:** `extra` `1` collects extra features for modeling (**WARNING:** takes MUCH longer to run), `0` does not collect extra features;  default is `0` 
* **OPTIONAL:** `--under_sample` `1` under samples dataset to make balanced classes, `0` keeps original class distribution;  default is `0`
* **OPTIONAL:** `--limit` `> 0` stops data collection and resizes output to match limit, `-1` means unlimited; default is `-1`

## Training Model
Once training dataset has been created, run [`train.py`](https://github.com/cpuguy96/StepCOVNet/blob/master/stepcovnet/training/train.py).
```.bash
cd stepcovnet/training
python train.py -i --input <string> -o --output <string> --multi <int> --extra <int> --under_sample <int> --lookback <int> --limit <int> --name <string> --pretrained_model <string>
``` 
* `-i` `--input` input directory path to training dataset
* `-o` `--output` output directory path to save model 
* **OPTIONAL:** `--multi` `1` uses STFTs with multiple `frame_size`, `0` uses STFTs single `frame_size`;  default is `0`
* **OPTIONAL:** `--extra` `1` use extra features for modeling, `0` does not use extra features;  default is `0` 
* **OPTIONAL:** `--under_sample` 1 uses under sampled balanced dataset, `0` uses original class distribution dataset; default is `0`
* **OPTIONAL:** `--lookback` `> 1` is uses timeseries based on `lookback` when modeling, `1` uses non timeseries; default is `1`
* **OPTIONAL:** `--limit` `> 0` limits the amount of training samples used during training, `-1` uses all the samples; default is `-1`
* **OPTIONAL:** `--name` name to give the finished model; default names model based on dataset used
* **OPTIONAL:** `--pretrained_model` input path to pretrained model  

## TODO
* Add more support for training arrow selection model
* End-to-end unit tests for all modules 


## Credits
* Inspiration from the paper [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
* Most of the source code derived from [musical-onset-efficient](https://github.com/ronggong/musical-onset-efficient)
