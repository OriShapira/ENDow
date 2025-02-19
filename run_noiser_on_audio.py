import sys
import os
import math
import time
import torch
import random 
import torchaudio
import numpy as np
from scipy import signal
import rir_generator as rir
from torch.utils.data import Dataset, DataLoader
import logging

import config


def load_wav(args, wav_path):
    wav, sr = torchaudio.load(wav_path)
    if sr != args['fs']:
        wav = torchaudio.transforms.Resample(sr, args['fs'], resampling_method='sinc_interpolation')(wav)
    return wav, args['fs']  # resampled wav (tensor) and sample rate

def normalize(x, eps=0.00001):
    max_val = x.abs().max()
    x_norm = x / (max_val + eps)
    return x_norm

def convolve_align(h, clean_sig):
    h = torch.tensor(h).T
    delay = np.argmax(abs(h))
    x_h = signal.fftconvolve(np.array(clean_sig.squeeze(0)), np.array(h).squeeze(0)) 
    x_h_align = x_h[delay:clean_sig.shape[1]+delay]
    reverb_signal = torch.from_numpy(x_h_align)
    reverb_signal = reverb_signal.unsqueeze(0)
    return reverb_signal

def create_noise(args):
    noise_folder_path = args['noise_folder_path']
    noise = None
    for _ in range(7):
        rnd_chunk_number = random.randint(0, len(os.listdir(noise_folder_path)) - 1)
        noise_filepath = os.path.join(noise_folder_path, os.listdir(noise_folder_path)[rnd_chunk_number])
        white_noise, _ = load_wav(args, noise_filepath)
        if white_noise.shape[0] > 1:
            white_noise = white_noise[0,:].unsqueeze(0)
        noise = white_noise if noise is None else torch.cat((noise, white_noise), dim=-1)
    return noise 

def add_noise(args, signal, noise_signal, snr):
    # make the noise the same length as the foreground signal
    if noise_signal.shape[1] > signal.shape[1]:
        white_noise = noise_signal[:,:signal.shape[1]]
    elif noise_signal.shape[1] < signal.shape[1]: # repeat noise to create a longer noise
        white_noise = torch.tile(noise_signal, (signal.shape[1],))[:signal.shape[1]]

    g = np.sqrt(10**(-snr/10) * np.std(np.array(signal))**2 / (args['eps'] + np.std(np.array(noise_signal))**2)) # Noise factor to get desired SNR
    noisy_signal = signal + g * white_noise
    
    return noisy_signal


class DatasetSignals(Dataset):
    def __init__(self, args, need_reverb=True, need_noise=True, save_signal=True, recreate_if_exists=False):
        self.args = args
        self.data_dir = args['data_dir']
        self.need_reverb = need_reverb
        self.need_noise = need_noise
        self.need_save_signal = save_signal
        self.all_filepaths_input = []
        for dirpath, _, filenames in os.walk(self.data_dir):
            for f in filenames:
                self.all_filepaths_input.append(os.path.abspath(os.path.join(dirpath, f)))
        self.all_signals = self.__get_processed_signals()

    def __len__(self):
        return len(self.all_filepaths_input)

    def __get_save_path(self, src_wav_filepath, is_reverbed, noise_snr):
        base_foldername_parts = [
            self.data_dir,
            '___'
            'reverb' if is_reverbed else 'noreverb',
            '__',
            f'noise_{noise_snr}' if noise_snr is not None else 'nonoise'
        ]
        base_foldername = ''.join(base_foldername_parts)
        transcript_name = os.path.basename(os.path.dirname(src_wav_filepath))
        new_filename = os.path.basename(src_wav_filepath)  # same as source filename
        save_path = os.path.join(base_foldername, transcript_name, new_filename)
        return save_path

    def __get_processed_signals(self):
        # get the paths of the folders that have noised signals based on the input signals:
        relevant_folderpaths = []
        data_foldername = os.path.basename(self.data_dir)
        data_folder_location = os.path.dirname(self.data_dir)
        for f in os.listdir(data_folder_location):
            if os.path.isdir(f) and f.startswith(data_foldername) and '___' in f:
                relevant_folderpaths.append(os.path.join(data_folder_location, f))

        # collect the processed signals from the relvant folders:
        processed_signals = {}
        for relevant_folderpath in relevant_folderpaths:
            for dirpath, _, filenames in os.walk(relevant_folderpath):
                transcript_name = os.path.basename(dirpath)
                for f in filenames:
                    save_filepath = os.path.abspath(os.path.join(dirpath, f))
                    orig_filepath = os.path.join(self.data_dir, transcript_name, f)
                    if orig_filepath not in processed_signals:
                        processed_signals[orig_filepath] = []
                    processed_signals[orig_filepath].append(save_filepath)
    
        return processed_signals

    def __getitem__(self, idx):
        wav_filepath = self.all_filepaths_input[idx]
        logging.info(f'File {wav_filepath}')
        try:
            if wav_filepath in self.all_signals and len(self.all_signals[wav_filepath]) == len(self.args['noise_snr_list']):
                logging.info(f'\tAlready found {wav_filepath}... skipping.')
            else:
                logging.info(f'\tProcessing {wav_filepath}')
                self.all_signals[wav_filepath] = []

                # reverberize the signal if needed:
                if self.need_reverb:
                    logging.info(f'\tFile {wav_filepath}: reverb')
                    signal, rir, sr = create_reverbed_signal(self.args, wav_filepath)
                else:
                    signal, sr = torchaudio.load(wav_filepath)

                # add background noise to the signal if needed, with different signal-to-noise ratios:
                if self.need_noise:
                    signals_info = []
                    noise_signal = create_noise(self.args)
                    for snr in self.args['noise_snr_list']:
                        logging.info(f'\tFile {wav_filepath}: noise {snr}')
                        noisy_sig = add_noise(self.args, signal, noise_signal, snr)
                        signals_info.append({'signal': noisy_sig, 'sr': sr, 'reverbed': self.need_reverb, 'noise_snr': snr})
                else:
                    signals_info = [{'signal': signal, 'sr': sr, 'reverbed': self.need_reverb, 'noise_snr': None}]

                # save the signals if needed:
                if self.need_save_signal:
                    for sig_info in signals_info:
                        save_path = self.__get_save_path(wav_filepath, sig_info['reverbed'], sig_info["noise_snr"])
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torchaudio.save(save_path, sig_info['signal'], sig_info['sr'])
                        self.all_signals[wav_filepath].append(save_path)
                        logging.info(f'\tSaved {wav_filepath} to {save_path}')
        except:
            logging.info(f'\tERROR: File {wav_filepath} skipped due to error')
            print(f'ERROR: File {wav_filepath} skipped due to error')
            self.all_signals[wav_filepath] = []

        return self.all_signals[wav_filepath]
    
    def get_item(self, idx):
        return self.__getitem__(idx)


class RIR_Generator:
    def __init__(self, args):
        self.args = args

    def __is_point_in_room(self, point, room_size):
        return self.args['dist_from_wall'] <= point[0] <= room_size[0]-self.args['dist_from_wall'] and self.args['dist_from_wall'] <= point[1] <= room_size[1]- self.args['dist_from_wall']

    def __random_point_with_distance(self, source_point, room_size, distance, num_attempts=30):
        for i in range(num_attempts):
            angle = random.uniform(0, 2 * math.pi)
            x = source_point[0] + distance * math.cos(angle)
            y = source_point[1] + distance * math.sin(angle)
            new_point = (x, y)
            if self.__is_point_in_room(new_point, room_size):
                return new_point
        return None

    def rand_room_missing_vals(self, room_measures=None, source_position=None, mic_location=None, rt60=False):
        rt60_range = self.args['rt60_range']
        dist_from_wall = self.args['dist_from_wall']
        min_dist, max_dist = self.args['mic_source_dist'][0], self.args['mic_source_dist'][1]
        min_w, max_w = self.args['room_dims'][0][0], self.args['room_dims'][0][1]
        min_l, max_l = self.args['room_dims'][1][0], self.args['room_dims'][1][1]
        min_h, max_h = self.args['room_dims'][2][0], self.args['room_dims'][2][1]

        # get a random room size if not provided:
        if room_measures is None:
            room_measures = np.array([random.uniform(min_w, max_w),random.uniform(min_l, max_l), random.uniform(min_h, max_h)])
        
        # get a radom position of the speaker if not provided:
        if source_position is None:
            source_position = np.array([random.uniform(dist_from_wall, room_measures[0]-dist_from_wall),
                                        random.uniform(dist_from_wall, room_measures[1]-dist_from_wall),
                                        random.uniform(0, room_measures[2])])

        # get a random point in the room for the microphone if not provided:
        mic_source_dist = random.uniform(min_dist, max_dist)
        while mic_location is None:
            mic_location = self.__random_point_with_distance(source_position, room_measures, mic_source_dist)
            if mic_location is None:
                mic_source_dist = mic_source_dist - 1
        mic_location = np.array([mic_location[0], mic_location[1], 3])

        # get a random rt60 if not provided:
        if rt60 is None:
            rt60 = random.uniform(rt60_range[0], rt60_range[1])
        
        #print('Room shape is {0:.2f}x{1:.2f}x{2:.2f}'.format(room_measures), 'source position is {0:.2f}x{1:.2f}x{2:.2f}'.format(source_position),
        #      'mic_location is {0:.2f}x{1:.2f}x{2:.2f}'.format(mic_location), 'rt60 is {0:.2f}x{1:.2f}x{2:.2f}'.format(rt60))

        return rt60, room_measures, mic_source_dist, source_position, mic_location

    # rir generator
    def generate_rir(self, room_measures, source_position, mic_location, rt60, fs, nsample):
        h = rir.generate(
            c = 340,                   # Sound velocity (m/s)
            fs = fs,                   # Sample frequency (samples/s)
            r = mic_location,          # Receiver position(s) [x y z] (m)
            s = source_position,       # Source position [x y z] (m)
            L = room_measures,         # Room dimensions [x y z] (m)
            reverberation_time = rt60, # Reverberation time (s)
            nsample = nsample,         # Number of output samples
        )
        return h
    

def create_reverbed_signal(args, signal_filepath, rt60=None, room_measures=None, 
                           mic_location=None, source_position=None, rir=None):
    clean_sig, sr = torchaudio.load(signal_filepath)

    if rir is None:
        rir_generator = RIR_Generator(args)
        
        if (args['center']):
            mean_value = torch.mean(clean_sig)  # Compute the mean value of the signal
            clean_sig = clean_sig - mean_value  # Center the signal at zero by subtracting the mean

        # get a random value for any of the missing values here:
        rt60, room_measures, mic_source_dist, source_position, mic_location = rir_generator.rand_room_missing_vals(room_measures=room_measures, 
                               source_position=source_position, 
                               mic_location=mic_location, rt60=rt60)
            
        n_samples = int(sr * rt60)
        rir = rir_generator.generate_rir(room_measures, source_position, mic_location, rt60, sr, nsample=n_samples)
        
        if args['save_rir']:
            rir_path = os.path.join(args['path_rirs'], f'rt60_{np.round(rt60, decimals=2)}_room_{np.round(room_measures, decimals=2)}_mic_source_dist_{np.round(mic_source_dist, decimals=2)}.wav')
            torchaudio.save(rir_path, normalize(torch.tensor(rir, dtype=torch.float32)).T, sr)
        
    # create the reverbed signal and align
    reverb_signal = convolve_align(rir, clean_sig)
    
    return reverb_signal, rir, sr


# This main function runs the noiser with a dataloader to parallelize the process, running much faster.
# For some reason, some instances fail using this method, in which case we can use `parallelize=False``,
# which runs the process serially, and does not fail.
def main(args, parallelize=True):
    start_time = time.time()

    dataset_signals = DatasetSignals(args, need_reverb=True, need_noise=True, save_signal=True)

    if parallelize:
        multiprocessing_dataloader = DataLoader(dataset_signals, batch_size=8, shuffle=False, 
                                                num_workers=args['num_workers'], pin_memory=args['pin_memory'])
        iterator = iter(multiprocessing_dataloader)
        i = 0
        while True:
            try:
                batch = next(iterator)
                i += 1
            except StopIteration:
                break
            except:
                print(f'ERROR {i}')
    else:
        for i in range(len(dataset_signals)):
            dataset_signals.get_item(i)

    gen_time = time.time() - start_time
    print(f"\nExecution time: {gen_time} seconds")


if __name__ == '__main__':
    # usage:
    # python3 run_noiser_on_audio.py <dataset_id> [--serial]
    # dataset_id: one of qmsum | qaconv | mrda
    # --serial: (optional) instead of running the noiser parallelized (much faster), run it serially
    #
    # Paths to set:
    #   BASE_PATH: the base path for this project
    #   NOISE_FOLDER_PATH: path to a folder with at least one audio file 
    #                      to use for bacground sounds to add for noising.
    BASE_PATH = config.BASE_PATH
    NOISE_FOLDER_PATH = config.NOISE_FOLDER_PATH

    assert len(sys.argv) >= 2, 'Error: You must pass in an argument <qmsum|qaconv|mrda>'
    dataset_id = sys.argv[1]  # qmsum | qaconv | mrda
    assert dataset_id in ['qmsum', 'qaconv', 'mrda'], 'Error: Dataset can be one of qmsum, qaconv or mrda.'

    logging.basicConfig(level=logging.INFO, 
                        filename=f"logfile_noiser_{dataset_id}", 
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    if len(sys.argv) > 2 and sys.argv[2] == '--serial':
        parallelize = False
    else:
        parallelize = True

    args = {
        'data_dir': f'/{BASE_PATH}/audio/{dataset_id}_test',
        'fs': 8000,
        'eps': 1e-6,
        'save_rir': False,
        'num_workers': 4,
        'pin_memory': False,
        # Reverb parameters
        'n_mics': 1,
        'n_speak': 1,
        'room_dims': [[2,10], [2,10], [3,3]],
        'rt60_range': [0.15, 1],
        'mic_source_dist': [2,2],
        'dist_from_wall': 0.5,
        'center': False,
        # noise paramaters
        'noise_folder_path': NOISE_FOLDER_PATH,
        'noise_snr_list': [-10, -5, 0, 5, 10]
    }
    main(args, parallelize=parallelize)