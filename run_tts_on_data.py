import sys
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
from tqdm import tqdm
import os
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize

import logging
import config


class DataTTS:
    def __init__(self):
        self.tts = TextToSpeech(use_deepspeed=True, kv_cache=True)
        self.preset = "ultra_fast"  # Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
        self.voice = 'emma' #'tom'  # Pick one of the voices
        self.voice_samples, self.conditioning_latents = load_voice(self.voice)

    def remove_markers_from_text(self, utterance):
        # Remove substrings within curly brackets
        cleaned_string = re.sub(r'\{.*?\}', '', utterance)
        # Remove substrings within box brackets
        cleaned_string = re.sub(r'\[.*?\]', '', cleaned_string)
        # Replace multiple spaces with a single space
        cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
        return cleaned_string
    
    def get_output_filepath_for_utterance(self, base_folderpath, conversation_name, utterance_id):
        pass
    
    def prepare_audio_files(self):
        pass
    
    def create_audio_file(self, utterance, outpath_wav, recreate_if_exists=False):
        if not os.path.exists(outpath_wav) or recreate_if_exists:
            os.makedirs(os.path.dirname(outpath_wav), exist_ok=True)
            try:
                gen = self.tts.tts_with_preset(utterance, voice_samples=self.voice_samples, 
                                               conditioning_latents=self.conditioning_latents,
                                               preset=self.preset)
                torchaudio.save(outpath_wav, gen.squeeze(0).cpu(), 24000)
            except:
                logging.info(f'Error: {utterance}')
                return None
        return outpath_wav



class DataTTS_QMSUM(DataTTS):
    def __init__(self, input_data_folder_path, output_folder_path):
        self.input_data_folder_path = input_data_folder_path
        self.output_folder_path = output_folder_path
        super().__init__()

    def get_output_filepath_for_utterance(self, base_folderpath, conversation_name, utterance_id):
        return os.path.join(base_folderpath, conversation_name, f'{conversation_name}_{utterance_id}.wav')

    def prepare_audio_files(self):
        for filename in tqdm(os.listdir(self.input_data_folder_path)):
            datum_filepath = os.path.join(data_folder_path, filename)
            with open(datum_filepath) as fIn:
                datum = json.load(fIn)
                for utt_idx, utt_info in tqdm(enumerate(datum['meeting_transcripts'])):
                    utt = self.remove_markers_from_text(utt_info['content']).strip()
                    for sent_idx, sent in enumerate(sent_tokenize(utt)):
                        if len(sent) >= 300:  # break up long sentences since the TTS model is limited to 400 tokens
                            words = word_tokenize(sent)
                            subsentences = [(' '.join(words[i:i+50]),  f'{utt_idx}_{sent_idx}_{i}') for i in range(0, len(words), 50)]
                        else:
                            subsentences = [(sent,  f'{utt_idx}_{sent_idx}')]

                        for subsent, subsent_id in subsentences:
                            outpath_wav = self.get_output_filepath_for_utterance(output_audio_folder_path, filename[:-5], subsent_id)
                            if not os.path.exists(outpath_wav) and len(subsent) > 0:
                                audio_filepath = self.create_audio_file(subsent, outpath_wav)
                                if audio_filepath is None:
                                    print(f'\n\n\n----------\nERROR with subsentence {outpath_wav} with len {len(subsent)} {len(word_tokenize(subsent))}\n----------')
                                    logging.info(f'{outpath_wav} Failed')
                                else:
                                    logging.info(f'{outpath_wav} Success')


class DataTTS_QACONV(DataTTS):
    def __init__(self, input_data_articles_json_path, input_data_json_path, input_domain_names_to_use, output_folder_path):
        self.input_data_articles_json_path = input_data_articles_json_path
        self.input_data_json_path = input_data_json_path
        self.input_domain_names_to_use = input_domain_names_to_use
        self.output_folder_path = output_folder_path

        # get the transcript IDs to use, as found in the test set:
        with open(input_data_json_path) as fIn:
            data_test = json.load(fIn)    
        self.transcript_ids_to_use = set()
        for i in data_test:
            transcript_id = i['article_full_id'][0]
            for input_domain_name_to_use in input_domain_names_to_use:
                if input_domain_name_to_use in transcript_id:
                    self.transcript_ids_to_use.add(transcript_id)

        # now get the actual transcripts with the IDs found in the test set:
        with open(input_data_articles_json_path) as fIn:
            self.data_transcripts = json.load(fIn)
            
        super().__init__()

    def get_output_filepath_for_utterance(self, base_folderpath, conversation_name, utterance_id):
        return os.path.join(base_folderpath, conversation_name, f'{utterance_id}.wav')

    def prepare_audio_files(self):
        for transcript_id in self.data_transcripts:
            if transcript_id in self.transcript_ids_to_use:
                for utt_info in tqdm(self.data_transcripts[transcript_id]):
                    utt_id = utt_info['id']
                    utt_speaker = utt_info['speaker']
                    utt_text = utt_info['text']
                    utt = self.remove_markers_from_text(utt_text).strip()
                    for sent_idx, sent in enumerate(sent_tokenize(utt)):
                        if len(sent) >= 300:  # break up long sentences since the TTS model is limited to 400 tokens
                            words = word_tokenize(sent)
                            subsentences = [(' '.join(words[i:i+50]),  f'{utt_id}_{sent_idx}_{i}') for i in range(0, len(words), 50)]
                        else:
                            subsentences = [(sent,  f'{utt_id}_{sent_idx}')]

                        for subsent, subsent_id in subsentences:
                            outpath_wav = self.get_output_filepath_for_utterance(output_audio_folder_path, transcript_id, subsent_id)
                            if not os.path.exists(outpath_wav) and len(subsent) > 0:
                                audio_filepath = self.create_audio_file(subsent, outpath_wav)
                                if audio_filepath is None:
                                    print(f'\n\n\n----------\nERROR with subsentence {outpath_wav} with len {len(subsent)} {len(word_tokenize(subsent))}\n----------')
                                    logging.info(f'{outpath_wav} Failed')
                                else:
                                    logging.info(f'{outpath_wav} Success')


class DataTTS_MRDA(DataTTS):
    def __init__(self, input_data_folder_path, output_folder_path):
        self.input_data_folder_path = input_data_folder_path
        self.output_folder_path = output_folder_path
        super().__init__()

    def get_output_filepath_for_utterance(self, base_folderpath, conversation_name, utterance_id):
        return os.path.join(base_folderpath, conversation_name, f'{conversation_name}_{utterance_id}.wav')

    def prepare_audio_files(self):
        for filename in tqdm(os.listdir(self.input_data_folder_path)):
            datum_filepath = os.path.join(self.input_data_folder_path, filename)
            with open(datum_filepath) as fIn:
                for utt_idx, utt_info_line in enumerate(fIn):
                    utt_parts = utt_info_line.strip().split('|')
                    utt_speaker = utt_parts[0]
                    utt_text = utt_parts[1]
                    utt_tag_basic = utt_parts[2]
                    utt_tag_general = utt_parts[3]
                    utt_tag_full = utt_parts[4]

                    outpath_wav = self.get_output_filepath_for_utterance(output_audio_folder_path, filename[:-4], utt_idx)
                    if not os.path.exists(outpath_wav):
                        audio_filepath = self.create_audio_file(utt_text, outpath_wav)
                        if audio_filepath is None:
                            print(f'\n\n\n----------\nERROR with subsentence {outpath_wav}\n----------')
                            logging.info(f'{outpath_wav} Failed')
                        else:
                            logging.info(f'{outpath_wav} Success')



if __name__ == '__main__':
    # usage:
    # python3 run_tts_on_data.py <dataset_id>
    # dataset_id: one of qmsum | qaconv | mrda
    #
    # Paths to set:
    #   BASE_PATH: the base path for this project
    BASE_PATH = config.BASE_PATH

    assert len(sys.argv) == 2, 'Error: You must pass in one argument <qmsum|qaconv|mrda>'
    dataset_id = sys.argv[1]  # qmsum | qaconv | mrda
    assert dataset_id in ['qmsum', 'qaconv', 'mrda'], 'Error: Dataset can be one of qmsum, qaconv or mrda.'

    logging.basicConfig(level=logging.INFO, filename=f"logfile_tts_{dataset_id}", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    if dataset_id == 'qmsum':
        data_folder_path = f'{BASE_PATH}/data/QMSum/test'
        output_audio_folder_path = f'{BASE_PATH}/audio/qmsum_test'
        data_tts_prep = DataTTS_QMSUM(data_folder_path, output_audio_folder_path)
    elif dataset_id == 'qaconv':
        data_articles_json_path = f'{BASE_PATH}/data/QAConv/article_full.json'
        data_test_json_path = f'{BASE_PATH}/data/QAConv/tst.json'
        output_audio_folder_path = f'{BASE_PATH}/audio/qaconv_test'
        domain_names_to_use = ['court', 'newidal'] # (there's a typo for newsdial)
        data_tts_prep = DataTTS_QACONV(data_articles_json_path, data_test_json_path, domain_names_to_use, output_audio_folder_path)
    elif dataset_id == 'mrda':
        data_folder_path = f'{BASE_PATH}/data/MRDA/test'
        output_audio_folder_path = f'{BASE_PATH}/audio/mrda_test'
        data_tts_prep = DataTTS_MRDA(data_folder_path, output_audio_folder_path)

    # create the audio files for the dataset transcripts:
    data_tts_prep.prepare_audio_files()