from collections import defaultdict
from os.path import join

import numpy as np
import resampy
import soundfile as sf

from stepcovnet.common import mel_features
from stepcovnet.common.audio_preprocessing import get_madmom_log_mels
from stepcovnet.common.utils import feature_reshape_up
from stepcovnet.encoder.BinaryArrowEncoder import BinaryArrowEncoder
from stepcovnet.encoder.LabelArrowEncoder import LabelArrowEncoder


def remove_out_of_range(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def feature_onset_phrase_label_sample_weights(frames_onset, mfcc, arrows, label_encoded_arrows, binary_encoded_arrows,
                                              num_arrow_types=4):
    # Depending on modeling results, it may be beneficial to clip all data from the first onset detected
    # to the last onset. This may affect how models interpret long periods of empty notes.
    frame_start = 0
    frame_end = mfcc.shape[0] - 1
    labels_dict = defaultdict(np.array)
    sample_weights_dict = defaultdict(np.array)
    arrows_dict = defaultdict(np.array)
    label_encoded_arrows_dict = defaultdict(np.array)
    binary_encoded_arrows_dict = defaultdict(np.array)

    for difficulty, onsets in frames_onset.items():
        # Might remove this functionality of poses issues with modeling
        frames_onsets_p25 = np.hstack((onsets - 1, onsets + 1))
        frames_onsets_p25 = np.sort(remove_out_of_range(frames_onsets_p25, frame_start, frame_end))
        onsets = remove_out_of_range(onsets, frame_start, frame_end)

        len_line = frame_end - frame_start + 1

        sample_weights = np.ones((len_line,))
        sample_weights[frames_onsets_p25 - frame_start] = 0.25
        sample_weights_dict[difficulty] = sample_weights.astype("float16")

        label = np.zeros((len_line,))
        label[onsets - frame_start] = 1
        label[frames_onsets_p25 - frame_start] = 1
        labels_dict[difficulty] = label.astype("int8")

        arrows_array = np.zeros((len_line, 4))
        label_encoded_arrows_array = np.zeros((len_line,))
        binary_encoded_arrows_array = np.zeros((len_line, 4 * num_arrow_types))

        arrows_list = arrows[difficulty]
        label_encoded_arrows_list = label_encoded_arrows[difficulty].reshape(-1)
        binary_encoded_arrows_list = binary_encoded_arrows[difficulty]
        i = 0
        for onset, arrow, label_encoded_arrow, binary_encoded_arrow in zip(onsets, arrows_list,
                                                                           label_encoded_arrows_list,
                                                                           binary_encoded_arrows_list):
            arrows_array[onset - frame_start] = arrow
            label_encoded_arrows_array[onset - frame_start] = label_encoded_arrow
            binary_encoded_arrows_array[onset - frame_start] = binary_encoded_arrow
            # This should be fine since timings should never be right next to each other
            if 2 * i < len(frames_onsets_p25):
                arrows_array[frames_onsets_p25[2 * i] - frame_start] = arrow
                label_encoded_arrows_array[frames_onsets_p25[2 * i] - frame_start] = label_encoded_arrow
                binary_encoded_arrows_array[frames_onsets_p25[2 * i] - frame_start] = binary_encoded_arrow
            if 2 * i + 1 < len(frames_onsets_p25):
                arrows_array[frames_onsets_p25[2 * i + 1] - frame_start] = arrow
                label_encoded_arrows_array[frames_onsets_p25[2 * i + 1] - frame_start] = label_encoded_arrow
                binary_encoded_arrows_array[frames_onsets_p25[2 * i + 1] - frame_start] = binary_encoded_arrow

            i += 1
        arrows_dict[difficulty] = arrows_array.astype("int8")
        label_encoded_arrows_dict[difficulty] = label_encoded_arrows_array.astype("int16")
        binary_encoded_arrows_dict[difficulty] = binary_encoded_arrows_array.astype("int8")

    mfcc_line = mfcc[frame_start:frame_end + 1, :]

    return mfcc_line, labels_dict, sample_weights_dict, arrows_dict, label_encoded_arrows_dict, binary_encoded_arrows_dict


def timings_parser(timing_file_path):
    """
    Read each line of timings file and parse arrows and timings
    :param timing_file_path: str - file name containing note timings
    :return: defaultdict - key: difficulty; value: list containing note arrows and timings
    """

    with open(timing_file_path, 'r') as file:
        data = defaultdict(dict)
        read_timings = False
        curr_difficulty = None
        label_encoder = LabelArrowEncoder()
        binary_encoder = BinaryArrowEncoder()
        for line in file.readlines():
            line = line.replace("\n", "")
            if line.startswith("NOTES"):
                read_timings = True
            elif read_timings:
                if line.startswith("DIFFICULTY"):
                    new_difficulty = line.split()[1].lower()
                    if new_difficulty in data:
                        raise ValueError("Same difficulty detected in the song data.")
                    curr_difficulty = new_difficulty
                elif curr_difficulty is not None:
                    arrows, timing = line.split(" ")[0:2]
                    label_encoded_arrows = label_encoder.encode(arrows)
                    binary_encoded_arrows = binary_encoder.encode(arrows)
                    data[curr_difficulty][float(timing)] = [np.array(list(arrows), dtype=int),
                                                            label_encoded_arrows, binary_encoded_arrows]
        return data


def get_fft_lengths(audio_sample_rate, window_length_secs=0.025, multi=False, num_multi_channels=3):
    fft_lengths = []
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    base_fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    if multi:  # if multi is used, normalization on the channel axis should be done
        if base_fft_length < 512:
            print("[WARN] Base fft length is less than 512. There may be data quality issues with smaller fft sizes.")
        num_fft_base_2 = int(np.log2(base_fft_length))
        for i in range(max(num_fft_base_2 - (num_multi_channels // 2), 0), num_fft_base_2):
            fft_lengths.append(2 ** i)
            num_multi_channels -= 1
        for i in range(num_fft_base_2, num_fft_base_2 + num_multi_channels):
            fft_lengths.append(2 ** i)
    else:
        fft_lengths = [base_fft_length]

    return fft_lengths, window_length_samples


def get_log_mels(audio_data, audio_data_sample_rate, config):
    # Convert to mono.
    if audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    else:
        audio_data = np.squeeze(audio_data)
    # Resample to the rate specified in config.
    if audio_data_sample_rate != config["SAMPLE_RATE"]:
        audio_data = resampy.resample(audio_data, audio_data_sample_rate, config["SAMPLE_RATE"])
    multi = True if config["NUM_CHANNELS"] > 1 else False
    fft_lengths, window_length_samples = get_fft_lengths(audio_sample_rate=config["SAMPLE_RATE"],
                                                         window_length_secs=config["STFT_WINDOW_LENGTH_SECONDS"],
                                                         multi=multi, num_multi_channels=config["NUM_MULTI_CHANNELS"])

    log_mels = []
    for fft_length in fft_lengths:
        # Compute log mel spectrogram features.
        log_mel = mel_features.log_mel_spectrogram(
            audio_data,
            audio_sample_rate=config["SAMPLE_RATE"],
            log_offset=np.spacing(1),
            hop_length_secs=config["STFT_HOP_LENGTH_SECONDS"],
            num_mel_bins=config["NUM_FREQ_BANDS"],
            lower_edge_hertz=config["MIN_FREQ"],
            upper_edge_hertz=config["MAX_FREQ"],
            fft_length=fft_length,
            window_length_samples=window_length_samples)

        # Create frame features.
        log_mel_frames = mel_features.frame(
            log_mel,
            window_length=config["NUM_TIME_BANDS"],
            hop_length=1)
        log_mels.append(log_mel_frames)

    if multi:
        return np.stack(log_mels, axis=3)
    else:
        return np.expand_dims(log_mels[0], axis=-1)


def get_audio_data(audio_file_path):
    """
    Return audio data and sample rate from an audio file
    :param audio_file_path:
    :return: audio_data (np.array): 2-d numpy array containing audio data (frames x channels)
             audio_data_sample_rate (int): audio file sample rate
    """
    return sf.read(audio_file_path, always_2d=True)


def convert_note_data(note_data, stft_hop_length_secs=0.01):
    # convert note timings into frame timings
    frames_onset = defaultdict(np.array)
    arrows_dict = defaultdict(np.array)
    label_encoded_arrows_dict = defaultdict(np.array)
    binary_encoded_arrows_dict = defaultdict(np.array)

    for difficulty, data in note_data.items():
        timings, arrows = list(data.keys()), list(data.values())
        frames_onset[difficulty] = np.array(np.around(np.array(timings) / stft_hop_length_secs), dtype=int)
        arrows_dict[difficulty] = np.array([arrow[0] for arrow in arrows], dtype=np.int8)
        label_encoded_arrows_dict[difficulty] = np.array([arrow[1] for arrow in arrows], dtype=np.int16)
        binary_encoded_arrows_dict[difficulty] = np.array([arrow[2] for arrow in arrows], dtype=np.int8)

    return frames_onset, arrows_dict, label_encoded_arrows_dict, binary_encoded_arrows_dict


def get_audio_features(wav_path, file_name, config):
    # Read audio data (needs to be a wav)
    audio_data, audio_data_sample_rate = get_audio_data(audio_file_path=join(wav_path, file_name + '.wav'))
    # Create log mel features
    log_mel_frames = get_log_mels(audio_data=audio_data, audio_data_sample_rate=audio_data_sample_rate, config=config)
    return log_mel_frames


def get_labels(note_data_path, file_name, config):
    # Read data from timings file
    note_data = timings_parser(timing_file_path=join(note_data_path, file_name + '.txt'))
    # Parse notes data to get onsets and arrows
    onsets, arrows, label_encoded_arrows, binary_encoded_arrows = \
        convert_note_data(note_data=note_data, stft_hop_length_secs=config["STFT_HOP_LENGTH_SECONDS"])
    return onsets, arrows, label_encoded_arrows, binary_encoded_arrows


def get_features_and_labels(wav_path, note_data_path, file_name, config):
    log_mel_frames = get_audio_features(wav_path, file_name, config)
    onsets, arrows, label_encoded_arrows, binary_encoded_arrows = get_labels(note_data_path, file_name, config)
    return log_mel_frames, onsets, arrows, label_encoded_arrows, binary_encoded_arrows


def get_features_and_labels_madmom(wav_path, note_data_path, file_name, multi, config):
    mfcc = get_madmom_log_mels(join(wav_path, file_name + '.wav'), multi, config=config)
    log_mel_frames = feature_reshape_up(feature=mfcc, num_freq_bands=config["NUM_FREQ_BANDS"],
                                        num_time_bands=config["NUM_TIME_BANDS"],
                                        num_channels=config["NUM_MULTI_CHANNELS"])
    note_data = timings_parser(join(note_data_path, file_name + '.txt'))
    onsets, arrows, label_encoded_arrows, binary_encoded_arrows = \
        convert_note_data(note_data=note_data, stft_hop_length_secs=config["STFT_HOP_LENGTH_SECONDS"])

    return log_mel_frames, onsets, arrows, label_encoded_arrows, binary_encoded_arrows
