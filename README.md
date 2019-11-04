# StepMania Note Generator
![header_example](https://github.com/cpuguy96/stepmania-note-generator/blob/master/imgs/header_example.gif)

## Requirements

* tensorflow-gpu>=2.0
* numpy
* madmom
* sklearn
* nltk
* librosa
* ffmpeg

## Running Audio to SM File Generator
### Currently only produces `.txt` files. Use [smfile_writer.py](https://github.com/jhaco/SMFile_Writer) to convert `.txt` to `.sm`
```bash
python stepmania_note_generator.py --input <string>  --output <string> -- scalers <string> --timing_model <string> --arrow_model <string> --overwrite <int>
```
* `--input` input directory path to audio files
* `--output` output directory path to `.sm` files
* `--scalers` input directory path to scalers used in training; **OPTIONAL:** default is `"training_data/"`
* `--timing_model` input directory path to beat timing model; **OPTIONAL:** default is `"models/retrained_timing_model.h5"`
* `--arrow_model` input directory path to arrow selection model; **OPTIONAL:** default is `"models/retrained_arrow_model.h5"`
* `--overwrite` `1` overwrites output `.sm` files, `0` skips if existing file found; **OPTIONAL:** default is `0`



## Creating Dataset
To create a training dataset, you need to parse the `.sm` files and convert sound files into `.wav` files: 
* [`smfile_parser.py`](https://github.com/jhaco/SMFile_Parser) should be used to parse the `.sm` files into `.txt` files. 
* [`wav_converter.py`](https://github.com/cpuguy96/stepmania-note-generator/blob/master/wrapper_scripts/wav_converter.py) can be used to convert the audio files into `.wav` files.

Once the parsed `.txt` files and `.wav` files are generated, place the `.wav` files into separate directories and run [`training_data_collection.py`](https://github.com/cpuguy96/stepmania-note-generator/blob/master/data_collection/training_data_collection.py).

```bash
python training_data_collection.py --audio <string>  --annotation <string> --output <string> --multi <int> --under_sample <int> --limit <int>
```
* `--audio` input directory path to `.wav` files
* `--annotation` input directory path to timing files
* `--output` output directory path to output dataset
* `--multi` 1 collects STFTs using `frame_size` of `[2048, 1024, 4096]`, `0` collects STFTs using `frame_size` of `[2048]`; **OPTIONAL:** default is `0`
* `--under_sample` 1 under samples dataset to make balanced classes, `0` keeps original class distribution; **OPTIONAL:** default is `0`
* `--limit` `>0` stops data collection and resizes output to match limit, `-1` means unlimited; **OPTIONAL:** default is `-1`

## Training Model
Once training dataset has been created, run [`train.py`](https://github.com/cpuguy96/stepmania-note-generator/blob/master/training_scripts/train.py).
```bash
python train.py --path_input <string> --path_output <string> --multi <int> --under_sample <int>
```
* `--path_input` input directory path to training dataset
* `--path_output` output directory path to save model 
* `--multi` 1 uses STFTs with multiple `frame_size`, `0` uses STFTs single `frame_size`; **OPTIONAL:** default is `0`
* `--under_sample` 1 uses under sampled balanced dataset, `0` uses original class distribution dataset; **OPTIONAL:** default is `0`

## TODO
* Add support for training arrow selection model
* Add single file input for wrapper and supporting scripts
* Add support for model renaming


## Credits
* Inspiration from the paper [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
* Most of the source code derived from [musical-onset-efficient](https://github.com/ronggong/musical-onset-efficient)
