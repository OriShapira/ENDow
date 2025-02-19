import logging
from datasets import Dataset
from transformers import pipeline
import evaluate
import os
import json
import re
import sys

import config


DEBUG = config.DEBUG
if DEBUG:
    DEBUG_DATASET_SIZE = 12
    SAVE_STATE_EVERY = 5
else:
    DEBUG_DATASET_SIZE = 0
    SAVE_STATE_EVERY = 100


metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")


def init_whisper(whisper_model_name, device):
    whisper_asr = pipeline("automatic-speech-recognition", model=whisper_model_name, device=device)
    whisper_norm = whisper_asr.tokenizer._normalize

    # helper function for normalization (punctuation, numbers, etc.)
    def normalize_func(batch):
        batch["norm_text"] = whisper_norm(get_text(batch))
        return batch
    
    return whisper_asr, normalize_func

# a helper function to filter out empty transcriptions:
def is_target_text_in_range(ref):
    r = ref.strip()
    if r == "ignore time segment in scoring":
        return False
    else:
        return r != ""
    
# helper function for getting the sample text
def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    else:
        raise ValueError(f"Sample: {sample.keys()} has no transcript.")

# helper function for cleaning the utterance text:
def remove_markers_from_text(utterance):
    cleaned_string = re.sub(r'\{.*?\}', '', utterance)  # Remove substrings within curly brackets
    cleaned_string = re.sub(r'\[.*?\]', '', cleaned_string)  # Remove substrings within box brackets
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)  # Replace multiple spaces with a single space
    return cleaned_string


class AudioDatasetQMSum(Dataset):
    def __init__(self, data_folderpath_audio, data_folderpath_transcript, sample_rate=16000):
        self.data_folderpath_audio = data_folderpath_audio
        self.data_folderpath_transcript = data_folderpath_transcript
        self.sample_rate = sample_rate
        self.utt_id_to_info, self.idx_to_utt_id = self._load_dataset()
        self.num_to_skip = 0
        self.num_items = len(self.utt_id_to_info)

    def _load_dataset(self):
        utt_id_to_info = {}  # utterance_id -> {'paths_audio': <list of path strs>, 'utterance': <str>}
        idx_to_utt_id = {}  # ordered index -> utterance id (to enable getitem function)

        # get the utterance texts:
        for filename in os.listdir(self.data_folderpath_transcript):
            transcript_filepath = os.path.join(self.data_folderpath_transcript, filename)
            with open(transcript_filepath) as fIn:
                datum = json.load(fIn)
                for utt_idx, utt_info in enumerate(datum['meeting_transcripts']):
                    utt_id = f'{filename[:-5]}_{utt_idx}'
                    utt = remove_markers_from_text(utt_info['content']).strip()
                    utt_id_to_info[utt_id] = {'paths_audio': [], 'utterance': utt}
                    idx_to_utt_id[len(idx_to_utt_id)] = utt_id

        # get the paths of the audio files (an utterance might have several audio files since it could have been split)
        audios = set()
        for dirpath, _, filenames in os.walk(self.data_folderpath_audio):
            for f in filenames:
                audio_filepath = os.path.abspath(os.path.join(dirpath, f))
                # Get the id of the utterance, e.g., Bed003_0_0.wav => Bed003_0, and Bed003_0_0___reverb__noise_-5_.wav => Bed003_0
                # such that the utterance id is the conversation id and the index of the utterance in the conversation.
                utt_id = '_'.join((f[:-4].split('___')[0]).split('_')[:-1])
                if utt_id not in utt_id_to_info:
                    utt_id = '_'.join(utt_id.split('_')[:-1])  # e.g., in case like Bed003_0_0_0.wav (extra _0 at the end)
                utt_id_to_info[utt_id]['paths_audio'].append(audio_filepath)

                audios.add(utt_id)

        print(f'utt_id_to_info - {len(utt_id_to_info)}')
        print(f'num_audios - {len(audios)}')

        return utt_id_to_info, idx_to_utt_id
    
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        idx_to_use = self.num_to_skip + idx

        utt_id = self.idx_to_utt_id[idx_to_use]
        datum = self.utt_id_to_info[utt_id]
        
        filepaths_audio = datum['paths_audio']
        utterance_text = datum['utterance']

        return {
            "audio_filepaths": filepaths_audio,
            "text": utterance_text,
            "id": utt_id
        }

    def take(self, n):
        self.num_items = n

    def get_item(self, idx):
        return self.__getitem__(idx)
    

class AudioDatasetQAConv(Dataset):
    def __init__(self, data_folderpath_audio, data_filepath_split, data_filepath_all_transcripts, sample_rate=16000):
        self.data_folderpath_audio = data_folderpath_audio
        self.data_filepath_split = data_filepath_split
        self.data_filepath_all_transcripts = data_filepath_all_transcripts
        self.sample_rate = sample_rate
        self.utt_id_to_info, self.idx_to_utt_id = self.__load_dataset()
        self.num_to_skip = 0
        self.num_items = len(self.utt_id_to_info)

    def __load_dataset(self):
        utt_id_to_info = {}  # utterance_id -> {'paths_audio': <list of path strs>, 'utterance': <str>}
        idx_to_utt_id = {}  # ordered index -> utterance id (to enable getitem function)

        # get the ids of the instances to use from the data split QAConv file:
        with open(self.data_filepath_split) as fIn:
            data_split = json.load(fIn)
        relevant_instances = set()
        for qa_instance in data_split:
            article_id = qa_instance['article_full_id'][0]
            if 'court' in article_id or 'newsidal' in article_id:
                relevant_instances.add(article_id)
        relevant_instances = sorted(list(relevant_instances))

        # get the utterance texts for the relevant instances:
        with open(self.data_filepath_all_transcripts) as fIn:
            data_transcripts = json.load(fIn)
        for transcript_id in relevant_instances:
            transcript_utts_info = data_transcripts[transcript_id]
            for utt_info in transcript_utts_info:
                utt_id = utt_info['id']
                utt = remove_markers_from_text(utt_info['text']).strip()
                utt_id_to_info[utt_id] = {'paths_audio': [], 'utterance': utt}
                idx_to_utt_id[len(idx_to_utt_id)] = utt_id

        # get the paths of the audio files (an utterance might have several audio files since it could have been split)
        audios = set()
        for dirpath, _, filenames in os.walk(self.data_folderpath_audio):
            for f in filenames:
                audio_filepath = os.path.abspath(os.path.join(dirpath, f))
                # Get the id of the utterance, e.g., court-04-1170-14033_6_0.wav => court-04-1170-14033, and court-04-1170-14033_6_0___reverb__noise_-5_.wav => court-04-1170-14033
                utt_id = f.split('_', 1)[0]
                utt_id_to_info[utt_id]['paths_audio'].append(audio_filepath)
                audios.add(utt_id)

        print(f'utt_id_to_info - {len(utt_id_to_info)}')
        print(f'num_audios - {len(audios)}')

        return utt_id_to_info, idx_to_utt_id

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        idx_to_use = self.num_to_skip + idx

        utt_id = self.idx_to_utt_id[idx_to_use]
        datum = self.utt_id_to_info[utt_id]
        
        filepaths_audio = datum['paths_audio']
        utterance_text = datum['utterance']

        return {
            "audio_filepaths": filepaths_audio,
            "text": utterance_text,
            "id": utt_id
        }

    def take(self, n):
        self.num_items = n

    def get_item(self, idx):
        return self.__getitem__(idx)


class AudioDatasetMRDA(Dataset):
    def __init__(self, data_folderpath_audio, data_folderpath_transcript, sample_rate=16000):
        self.data_folderpath_audio = data_folderpath_audio
        self.data_folderpath_transcript = data_folderpath_transcript
        self.sample_rate = sample_rate
        self.utt_id_to_info, self.idx_to_utt_id = self.__load_dataset()
        self.num_to_skip = 0
        self.num_items = len(self.utt_id_to_info)

    def __load_dataset(self):
        utt_id_to_info = {}  # utterance_id -> {'path_audio': <path_to_audio_file>, 'utterance': <str>}
        idx_to_utt_id = {}  # ordered index -> utterance id (to enable getitem function)

        # get the utterance texts:
        for filename in os.listdir(self.data_folderpath_transcript):
            transcript_filepath = os.path.join(self.data_folderpath_transcript, filename)
            with open(transcript_filepath) as fIn:
                for utt_idx, line in enumerate(fIn):
                    utt_info = line.strip().split('|')
                    utt_id = f'{filename[:-4]}_{utt_idx}'
                    utt_txt = utt_info[1].strip()
                    utt_id_to_info[utt_id] = {'path_audio': None, 'utterance': utt_txt}
                    idx_to_utt_id[len(idx_to_utt_id)] = utt_id

        # get the path to the audio file for each utterance:
        audios = set()
        for dirpath, _, filenames in os.walk(self.data_folderpath_audio):
            for f in filenames:
                audio_filepath = os.path.abspath(os.path.join(dirpath, f))
                utt_id = f[:-4]
                utt_id_to_info[utt_id]['path_audio'] = audio_filepath
                audios.add(utt_id)

        print(f'utt_id_to_info - {len(utt_id_to_info)}')
        print(f'num_audios - {len(audios)}')

        return utt_id_to_info, idx_to_utt_id

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        idx_to_use = self.num_to_skip + idx

        utt_id = self.idx_to_utt_id[idx_to_use]
        datum = self.utt_id_to_info[utt_id]
        
        filepath_audio = datum['path_audio']
        utterance_text = datum['utterance']
        
        return {
            "audio_filepath": filepath_audio,
            "text": utterance_text,
            "id": utt_id
        }

    def take(self, n):
        self.num_items = n

    def get_item(self, idx):
        return self.__getitem__(idx)



def load_datasets_qmsum(base_dir):
    logging.info('Loading Clean QMSum test data')
    qmsum_clean_test = AudioDatasetQMSum(f'{base_dir}/audio/qmsum_test', f'{base_dir}/data/QMSum/test')
    out_base_folderpath = f'{base_dir}/transcripts/qmsum_test'

    logging.info('Loading Noised QMSum test data - reberb -10')
    qmsum_noised_test_reverb_m10 = AudioDatasetQMSum(f'{base_dir}/audio/qmsum_test___reverb__noise_-10', 
                                                     f'{base_dir}/data/QMSum/test')
    out_base_folderpath_reverb_m10 = f'{base_dir}/transcripts/qmsum_test___reverb__noise_-10'

    logging.info('Loading Noised QMSum test data - reberb -5')
    qmsum_noised_test_reverb_m5 = AudioDatasetQMSum(f'{base_dir}/audio/qmsum_test___reverb__noise_-5', 
                                                    f'{base_dir}/data/QMSum/test')
    out_base_folderpath_reverb_m5 = f'{base_dir}/transcripts/qmsum_test___reverb__noise_-5'

    logging.info('Loading Noised QMSum test data - reberb 0')
    qmsum_noised_test_reverb_0 = AudioDatasetQMSum(f'{base_dir}/audio/qmsum_test___reverb__noise_0', 
                                                   f'{base_dir}/data/QMSum/test')
    out_base_folderpath_reverb_0 = f'{base_dir}/transcripts/qmsum_test___reverb__noise_0'

    logging.info('Loading Noised QMSum test data - reberb 5')
    qmsum_noised_test_reverb_5 = AudioDatasetQMSum(f'{base_dir}/audio/qmsum_test___reverb__noise_5', 
                                                   f'{base_dir}/data/QMSum/test')
    out_base_folderpath_reverb_5 = f'{base_dir}/transcripts/qmsum_test___reverb__noise_5'

    logging.info('Loading Noised QMSum test data - reberb 10')
    qmsum_noised_test_reverb_10 = AudioDatasetQMSum(f'{base_dir}/audio/qmsum_test___reverb__noise_10', 
                                                    f'{base_dir}/data/QMSum/test')
    out_base_folderpath_reverb_10 = f'{base_dir}/transcripts/qmsum_test___reverb__noise_10'

    datasets_to_use = {
        "qmsum_clean_test": {'ds': qmsum_clean_test, 'out_path': out_base_folderpath},
        "qmsum_noised_test_reverb_m10": {'ds': qmsum_noised_test_reverb_m10, 'out_path': out_base_folderpath_reverb_m10},
        "qmsum_noised_test_reverb_m5": {'ds': qmsum_noised_test_reverb_m5, 'out_path': out_base_folderpath_reverb_m5},
        "qmsum_noised_test_reverb_0": {'ds': qmsum_noised_test_reverb_0, 'out_path': out_base_folderpath_reverb_0},
        "qmsum_noised_test_reverb_5": {'ds': qmsum_noised_test_reverb_5, 'out_path': out_base_folderpath_reverb_5},
        "qmsum_noised_test_reverb_10": {'ds': qmsum_noised_test_reverb_10, 'out_path': out_base_folderpath_reverb_10}
    }
    
    for dn in datasets_to_use:
        # only for debugging, restricts the number of rows:
        if DEBUG_DATASET_SIZE > 0:
            datasets_to_use[dn]['ds'].take(DEBUG_DATASET_SIZE)

    return datasets_to_use


def load_datasets_qaconv(base_dir, subsets_to_load):
    # datasets_to_load is a list of data to use. it can include 'original', 'm10', 'm5', '0', '5', '10'

    datasets_to_use = {}

    data_filepath_split = f'{base_dir}/data/QAConv/tst.json'
    data_filepath_all_transcripts = f'{base_dir}/data/QAConv/article_full.json'

    if 'original' in subsets_to_load:
        logging.info('Loading Clean QAConv test data')
        qaconv_clean_test = AudioDatasetQAConv(f'{base_dir}/audio/qaconv_test', 
                                               data_filepath_split, data_filepath_all_transcripts)
        out_base_folderpath = f'{base_dir}/transcripts/qaconv_test'
        datasets_to_use["qaconv_clean_test"] = {'ds': qaconv_clean_test, 'out_path': out_base_folderpath}

    if 'm10' in subsets_to_load:
        logging.info('Loading Noised QAConv test data - reberb -10')
        qaconv_noised_test_reverb_m10 = \
            AudioDatasetQAConv(f'{base_dir}/audio/qaconv_test___reverb__noise_-10', 
                               data_filepath_split, data_filepath_all_transcripts)
        out_base_folderpath_reverb_m10 = f'{base_dir}/transcripts/qaconv_test___reverb__noise_-10'
        datasets_to_use["qaconv_noised_test_reverb_m10"] = {'ds': qaconv_noised_test_reverb_m10, 'out_path': out_base_folderpath_reverb_m10}

    if 'm5' in subsets_to_load:
        logging.info('Loading Noised QAConv test data - reberb -5')
        qaconv_noised_test_reverb_m5 = \
            AudioDatasetQAConv(f'{base_dir}/audio/qaconv_test___reverb__noise_-5',
                               data_filepath_split, data_filepath_all_transcripts)
        out_base_folderpath_reverb_m5 = f'{base_dir}/transcripts/qaconv_test___reverb__noise_-5'
        datasets_to_use["qaconv_noised_test_reverb_m5"] = {'ds': qaconv_noised_test_reverb_m5, 'out_path': out_base_folderpath_reverb_m5}

    if '0' in subsets_to_load:
        logging.info('Loading Noised QAConv test data - reberb 0')
        qaconv_noised_test_reverb_0 = \
            AudioDatasetQAConv(f'{base_dir}/audio/qaconv_test___reverb__noise_0', 
                               data_filepath_split, data_filepath_all_transcripts)
        out_base_folderpath_reverb_0 = f'{base_dir}/transcripts/qaconv_test___reverb__noise_0'
        datasets_to_use["qaconv_noised_test_reverb_0"] = {'ds': qaconv_noised_test_reverb_0, 'out_path': out_base_folderpath_reverb_0}

    if '5' in subsets_to_load:
        logging.info('Loading Noised QAConv test data - reberb 5')
        qaconv_noised_test_reverb_5 = \
            AudioDatasetQAConv(f'{base_dir}/audio/qaconv_test___reverb__noise_5', 
                               data_filepath_split, data_filepath_all_transcripts)
        out_base_folderpath_reverb_5 = f'{base_dir}/transcripts/qaconv_test___reverb__noise_5'
        datasets_to_use["qaconv_noised_test_reverb_5"] = {'ds': qaconv_noised_test_reverb_5, 'out_path': out_base_folderpath_reverb_5}

    if '10' in subsets_to_load:
        logging.info('Loading Noised QAConv test data - reberb 10')
        qaconv_noised_test_reverb_10 = \
            AudioDatasetQAConv(f'{base_dir}/audio/qaconv_test___reverb__noise_10',
                               data_filepath_split, data_filepath_all_transcripts)
        out_base_folderpath_reverb_10 = f'{base_dir}/transcripts/qaconv_test___reverb__noise_10'
        datasets_to_use["qaconv_noised_test_reverb_10"] = {'ds': qaconv_noised_test_reverb_10, 'out_path': out_base_folderpath_reverb_10}
        

    for dn in datasets_to_use:
        # only for debugging, restricts the number of rows:
        if DEBUG_DATASET_SIZE > 0:
            datasets_to_use[dn]['ds'].take(DEBUG_DATASET_SIZE)

    return datasets_to_use


def load_datasets_mrda(base_dir):
    logging.info('Loading Clean MRDA test data')
    mrda_clean_test = AudioDatasetMRDA(f'{base_dir}/audio/mrda_test', f'{base_dir}/data/MRDA/test')
    out_base_folderpath = f'{base_dir}/transcripts/mrda_test'

    logging.info('Loading Noised MRDA test data - reberb -10')
    mrda_noised_test_reverb_m10 = AudioDatasetMRDA(f'{base_dir}/audio/mrda_test___reverb__noise_-10',
                                                   f'{base_dir}/data/MRDA/test')
    out_base_folderpath_reverb_m10 = f'{base_dir}/transcripts/mrda_test___reverb__noise_-10'

    logging.info('Loading Noised MRDA test data - reberb -5')
    mrda_noised_test_reverb_m5 = AudioDatasetMRDA(f'{base_dir}/audio/mrda_test___reverb__noise_-5',
                                                  f'{base_dir}/data/MRDA/test')
    out_base_folderpath_reverb_m5 = f'{base_dir}/transcripts/mrda_test___reverb__noise_-5'

    logging.info('Loading Noised MRDA test data - reberb 0')
    mrda_noised_test_reverb_0 = AudioDatasetMRDA(f'{base_dir}/audio/mrda_test___reverb__noise_0',
                                                 f'{base_dir}/data/MRDA/test')
    out_base_folderpath_reverb_0 = f'{base_dir}/transcripts/mrda_test___reverb__noise_0'

    logging.info('Loading Noised MRDA test data - reberb 5')
    mrda_noised_test_reverb_5 = AudioDatasetMRDA(f'{base_dir}/audio/mrda_test___reverb__noise_5',
                                                 f'{base_dir}/data/MRDA/test')
    out_base_folderpath_reverb_5 = f'{base_dir}/transcripts/mrda_test___reverb__noise_5'

    logging.info('Loading Noised MRDA test data - reberb 10')
    mrda_noised_test_reverb_10 = AudioDatasetMRDA(f'{base_dir}/audio/mrda_test___reverb__noise_10',
                                                  f'{base_dir}/data/MRDA/test')
    out_base_folderpath_reverb_10 = f'{base_dir}/transcripts/mrda_test___reverb__noise_10'

    datasets_to_use = {
        "mrda_clean_test": {'ds': mrda_clean_test, 'out_path': out_base_folderpath},
        "mrda_noised_test_reverb_m10": {'ds': mrda_noised_test_reverb_m10, 'out_path': out_base_folderpath_reverb_m10},
        "mrda_noised_test_reverb_m5": {'ds': mrda_noised_test_reverb_m5, 'out_path': out_base_folderpath_reverb_m5},
        "mrda_noised_test_reverb_0": {'ds': mrda_noised_test_reverb_0, 'out_path': out_base_folderpath_reverb_0},
        "mrda_noised_test_reverb_5": {'ds': mrda_noised_test_reverb_5, 'out_path': out_base_folderpath_reverb_5},
        "mrda_noised_test_reverb_10": {'ds': mrda_noised_test_reverb_10, 'out_path': out_base_folderpath_reverb_10}
    }
    
    for dn in datasets_to_use:
        # only for debugging, restricts the number of rows:
        if DEBUG_DATASET_SIZE > 0:
            datasets_to_use[dn]['ds'].take(DEBUG_DATASET_SIZE)

    return datasets_to_use


def load_info_for_inference(base_dir, model_name, dataset_name):
    # get the first instances index to start inference from (for the specified model and dataset pair)
    last_index_folder_path = os.path.join(base_dir, 'temp', 'last_index')
    last_index_file_path = os.path.join(last_index_folder_path, f'{model_name}__{dataset_name}.txt')
    if not os.path.exists(last_index_folder_path):
        os.makedirs(last_index_folder_path)
    last_index = -1
    if os.path.exists(last_index_file_path):
        with open(last_index_file_path, 'r') as fIn:
            for line in fIn:
                line = line.strip()
                if line != '':
                    if 'END' in line:
                        last_index = None
                    else:
                        last_index = int(line.strip())
                break
    first_datum_index_to_use = last_index + 1 if last_index != None else None

    return first_datum_index_to_use


def dump_info_for_inference(base_dir, model_name, dataset_name, list_of_new_instances, last_instance_idx, is_end_of_dataset=False):
    # dump the new instances created
    out_file_path = os.path.join(base_dir, f'{model_name}__{dataset_name}.jsonl')
    with open(out_file_path, 'a') as fOut:
        fOut.write('\n'.join([json.dumps(d) for d in list_of_new_instances]) + '\n')

    # dump the last index processed
    last_index_file_path = os.path.join(base_dir, 'temp', 'last_index', f'{model_name}__{dataset_name}.txt')
    with open(last_index_file_path, 'w') as fOut:
        fOut.write(str(last_instance_idx))
        if is_end_of_dataset:
            fOut.write(' -- END OF DATASET')

def normalize_utterance(text, whisper_normalizer):
    text = whisper_normalizer(text)
    replacements = [
        ('okay', 'ok'),
        (" 'm", "'m"),
        (" 'll", "'ll"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" n't", "n't")
    ]
    for r1, r2 in replacements:
        text = text.replace(r1, r2)
    return text


def main(dataset_id, base_dir_path, device, subsets_to_process_qaconv=None):
    logging.info(f'Loading datasets...')
    if dataset_id == 'qmsum':
        datasets_to_use = load_datasets_qmsum(base_dir_path)
    elif dataset_id == 'qaconv':
        datasets_to_use = load_datasets_qaconv(base_dir_path, subsets_to_process_qaconv)
    elif dataset_id == 'mrda':
        datasets_to_use = load_datasets_mrda(base_dir_path)
    logging.info(f'Done loading datasets')
    whisper_model_name = 'whisper_small'
    whisper_model_path = 'openai/whisper-small.en'

    logging.info(f'Loading model')
    whisper_asr, normalize_func = init_whisper(whisper_model_path, device)

    logging.info(f'Starting inference')
    # loop over all the datasets in the ESB benchmark
    for dataset_name, dataset_info in datasets_to_use.items():
        logging.info(f'\tOn dataset {dataset_name}')

        dataset = dataset_info['ds']
        out_base_folder_path = dataset_info['out_path']
        
        # initialize stuff
        first_datum_index_to_use = load_info_for_inference(out_base_folder_path, whisper_model_name, dataset_name)
        if first_datum_index_to_use == None:
            logging.info(f'\tThe dataset was already processed in full. Skipping.')
            continue

        # run streamed inference
        logging.info(f'\tLooping over data, starting from index {first_datum_index_to_use}')
        new_outputs = []
        num_instances_processed = 0
        num_instances_total = first_datum_index_to_use
        #for idx, datum in enumerate(dataset, first_datum_index_to_use):
        for idx in range(first_datum_index_to_use, len(dataset)):
            datum = dataset.get_item(idx)
            num_instances_total += 1
            if idx >= first_datum_index_to_use:
                
                if 'audio_filepath' in datum:  # mrda has one audio per datum
                    if datum['audio_filepath'] is not None:
                        out = whisper_asr(datum['audio_filepath'])
                        out_text = out['text']
                    else:
                        out_text = ''
                elif 'audio_filepaths' in datum:  # the other datasets can have several audios per datum
                    out = whisper_asr(datum['audio_filepaths'])
                    out_text = ' '.join([o['text'] for o in out])

                pred = normalize_utterance(out_text, whisper_asr.tokenizer._normalize)
                ref = normalize_utterance(datum["text"], whisper_asr.tokenizer._normalize)

                if len(pred) == 0 or len(ref) == 0:
                    wer = 0
                    cer = 0
                else:
                    wer = metric_wer.compute(references=[ref], predictions=[pred])
                    cer = metric_cer.compute(references=[ref], predictions=[pred])
                new_outputs.append({'id': datum["id"], 'ref': datum["text"], 'pred': out_text, 'idx': idx, 'wer': wer, 'cer': cer})
                if len(new_outputs) == SAVE_STATE_EVERY:
                    logging.info(f'\t\tSaving state at idx {idx}')
                    dump_info_for_inference(out_base_folder_path, whisper_model_name, dataset_name, new_outputs, idx)
                    new_outputs = []
                num_instances_processed += 1

        
        logging.info(f'\t\tSaving state at idx {idx} (last instance)')
        dump_info_for_inference(out_base_folder_path, whisper_model_name, dataset_name, 
                                new_outputs, idx, is_end_of_dataset=True)
            
        logging.info(f'\tSummary: Processed {num_instances_processed} of {num_instances_total} instances for {dataset_name} with model {whisper_model_name}')
    
    logging.info('Done!')
    



if __name__ == '__main__':
    # usage:
    # python3 run_stt_on_audio.py <dataset_id> [sets_to_process_space_separated]
    # dataset_id: one of qmsum | qaconv | mrda
    # sets_to_process: (optional for qaconv) the subsets to process (any of original | m10 | m5 | 0 | 5 | 10)
    #
    # Variables to set:
    #   BASE_PATH: the base path for this project
    BASE_PATH = config.BASE_PATH
    MY_DEVICE = config.MY_DEVICE

    assert len(sys.argv) >= 2, 'Error: You must pass in the first argument <qmsum|qaconv|mrda>'
    dataset_id = sys.argv[1]  # qmsum | qaconv | mrda
    assert dataset_id in ['qmsum', 'qaconv', 'mrda'], 'Error: Dataset can be one of qmsum, qaconv or mrda.'

    logging.basicConfig(level=logging.INFO, filename=f"logfile_stt_{dataset_id}", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger(f"logfile_stt_{dataset_id}")

    # the qaconv dataset can also get a list of subsets to process (since it is a mich larger dataset):
    if dataset_id == 'qaconv':
        if len(sys.argv) > 2:
            subsets_to_process = sys.argv[2:]
        else:
            subsets_to_process = ['original', 'm10', 'm5', '0', '5', '10']
    else:
        subsets_to_process = None

    try:
        main(dataset_id, BASE_PATH, MY_DEVICE, subsets_to_process_qaconv=subsets_to_process)
    except Exception as e:
        logger.exception("An unexpected error occurred")