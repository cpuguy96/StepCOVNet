# StepMania Note Generator

## Requirements

* tensorflow-gpu>=2.0
* numpy
* madmom
* sklearn
* nltk
* librosa
* ffmpeg

## Creating Dataset
To create a training dataset, you need to parse the `.sm` files and convert sound files into `.wav` files: 
* [`smfile_parser.py`](https://github.com/jhaco/SMFile_Parser) should be used to parse the `.sm` files into `.txt` files. 
* [`wav_converter.py`](https://github.com/cpuguy96/stepmania-note-generator/blob/master/training_set_feature_extraction/wav_converter.py) can be used to convert the audio files into `.wav` files.

Once the parsed `.txt` files and `.wav` files are generated, place the `.wav` files into separate directories and run [`training_data_collection.py`](https://github.com/cpuguy96/stepmania-note-generator/blob/master/training_set_feature_extraction/training_data_collection.py).

```bash
python training_data_collection.py --audio <string>  --annotation <string> --output <string> --multi <int> --under_sample <int> --limit <int>
```
* `--audio` directory path to `.wav` files
* `--annotation` directory path to `.wav` files
* `--output` directory path to `.wav` files
* `--multi` 1 collects STFTs using `frame_size` of `[2048, 1024, 4096]`, `0` collects STFTs using `frame_size` of `[2048]`; _default_ `0`
* `--under_sample` 1 undersamples dataset to make balanced classes, `0` keeps original class distribution; _default_ `0`
* `--limit` `>=0` stops data collection and resizes output to match limit, `-1` means unlimited; _default_ `-1`

## Training Model
Once training dataset has been created, run [`train.py`](https://github.com/cpuguy96/stepmania-note-generator/blob/master/training_scripts/train.py).
```bash
python train.py --path_input <string> --path_output <string> --multi <int> --under_sample <int>
```
* `--path_input` directory path to training dataset
* `--path_output` directory path to save model 
* `--multi` 1 uses STFTs with multiple `frame_size`, `0` uses STFTs single `frame_size`; _default_ `0`
* `--under_sample` 1 uses under sampled balanced dataset, `0` uses original class distribution dataset; _default_ `0`

```
TODO: Add training arrow selection model
```

## Credits
* Inspiration from the paper [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
* Most of the source code derived from [musical-onset-efficient](https://github.com/ronggong/musical-onset-efficient)
