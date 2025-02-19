import sys
import os
import json
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score, accuracy_score

import config
from generation_models import GenModelMistral7BInstruct, GenModelLlama3Instruct, GenModelLlama3_1Instruct, GenModelGPT4oMini


class ProcessMRDA:

    def __init__(self, model,
                 input_mrda_data_folderpath, 
                 input_mrda_data_transcript_filepath=None,
                 output_filepath=None, num_utterances_per_conversation=100, debug=False):
        '''
        model_class: a class based on GenModelBase which will be used to process MRDA
        input_mrda_data_folderpath: the path to the MRDA data split directory
        input_mrda_data_transcript_filepath: if there is an alternative MRDA transcript file (e.g., noised)
            set the path to the file. This will be used for summarizing and scoring instead of the original
            MRDA transcript from input_mrda_data_folderpath. Otherwise set this with None.
        output_filepath: where to output the summaries and scores
        '''
        self.debug = debug
        if self.debug:
            print('DEBUG MODE')
        self.data_path = input_mrda_data_folderpath
        self.data_transcript_filepath = input_mrda_data_transcript_filepath
        self.output_filepath = output_filepath
        self.model = model
        self.data = self.__load_data()
        self.num_utterances_per_conversation = num_utterances_per_conversation
        
    def __load_alternative_transcript_file(self):
        conversation_id_to_utterances = {}
        if self.data_transcript_filepath is not None:
            with open(self.data_transcript_filepath, 'r') as fIn:
                for line in fIn:
                    datum = json.loads(line.strip())
                    utt_id = datum['id']
                    conversation_id = '_'.join(utt_id.split('_')[:-1])
                    utt_idx = int(utt_id.split('_')[-1])
                    utt_text = datum['pred']
                    if conversation_id not in conversation_id_to_utterances:
                        conversation_id_to_utterances[conversation_id] = {}
                    conversation_id_to_utterances[conversation_id][utt_idx] = utt_text
        return conversation_id_to_utterances  # conversation_id -> utterance_idx -> utterance_text

    def __load_data(self):
        alternative_transcripts = self.__load_alternative_transcript_file()

        data = {}  # conversation_id -> [{'idx': <int>, 'id': <str>, 'text': <str>, 'speaker': <str>, 'tag_basic': <str>, 'tag_general': <str>, 'tag_full': <str>}]
        for filename in os.listdir(self.data_path):
            filepath = os.path.join(self.data_path, filename)
            conversation_id = filename[:-4]
            data[conversation_id] = []
            with open(filepath, 'r') as fIn:
                for utt_idx, line in enumerate(fIn):
                    utt_info = line.split('|')
                    utt_id = f'{filename[:-4]}_{utt_idx}'
                    utt_txt_orig = utt_info[1].strip()
                    utt_speaker = utt_info[0].strip()
                    tag_basic = utt_info[2].strip()
                    tag_general = utt_info[3].strip()
                    tag_full = utt_info[4].strip()

                    # if there is an alternative trancript to use, use the utterance from there:
                    if conversation_id in alternative_transcripts:
                        if utt_idx in alternative_transcripts[conversation_id]:
                            utt = alternative_transcripts[conversation_id][utt_idx]
                        else:
                            utt = ''
                    else:  # otherwise use the utterance from the original MRDA data
                        utt = utt_txt_orig

                    data[conversation_id].append({'id': utt_id, 'idx': utt_idx, 'text_original': utt_txt_orig, 'text': utt, 'speaker': utt_speaker,
                                                  'tag_basic': tag_basic, 'tag_general': tag_general, 'tag_full': tag_full})
        return data

    def __compute_overall_scores(self, gold_labels, predicted_labels, num_samples=1000, confidence=0.95):
        """
        Compute F1 macro and accuracy with confidence intervals (using bootstrapping).

        Parameters:
            gold_labels (List[int]): The true labels for the classification task.
            predicted_labels (List[int]): The predicted labels from the model.
            num_samples (int): Number of bootstrap samples to generate (default 1000).
            confidence (float): The desired confidence level (default: 0.95).
        
        Returns:
            dictioanry of metrics with there confidence intervals {'f1_macro': {'score': <float>, 'confidence_interval_95': (<float>, <float>)},
                                                                'accuracy': {'score': <float>, 'confidence_interval_95': (<float>, <float>)},
                                                                'n': <int>}
        """

        def compute_f1(gold_labels, predicted_labels):
            return f1_score(gold_labels, predicted_labels, labels=['s', 'b', 'fh', 'qy', '%', 'fg', 'qw', 'h', 'qrr', 'qh', 'qr', 'qo'], average='macro')
        
        def compute_accuracy(gold_labels, predicted_labels):
            return accuracy_score(gold_labels, predicted_labels)

        # Ensure labels are NumPy arrays for easier indexing
        gold_labels = np.array(gold_labels)
        predicted_labels = np.array(predicted_labels)

        # Calculate the F1 score and accuracy for each bootstrap sample
        bootstrap_f1_scores = []
        bootstrap_acc_scores = []
        n = len(gold_labels)

        for _ in range(num_samples):
            indices = np.random.choice(range(n), size=n, replace=True)
            bootstrap_gold = gold_labels[indices]
            bootstrap_pred = predicted_labels[indices]
            score_f1 = compute_f1(bootstrap_gold, bootstrap_pred)
            score_acc = compute_accuracy(bootstrap_gold, bootstrap_pred)
            bootstrap_f1_scores.append(score_f1)
            bootstrap_acc_scores.append(score_acc)

        # Compute the confidence interval
        lower_bound_f1 = np.percentile(bootstrap_f1_scores, (1 - confidence) / 2 * 100)
        upper_bound_f1 = np.percentile(bootstrap_f1_scores, (1 + confidence) / 2 * 100)
        lower_bound_acc = np.percentile(bootstrap_acc_scores, (1 - confidence) / 2 * 100)
        upper_bound_acc = np.percentile(bootstrap_acc_scores, (1 + confidence) / 2 * 100)

        overall_f1 = compute_f1(gold_labels, predicted_labels)
        overall_acc = compute_accuracy(gold_labels, predicted_labels)

        overall_scores = {'f1_macro': {'score': float(overall_f1), 'confidence_interval_95': (float(lower_bound_f1), float(upper_bound_f1))},
                        'accuracy': {'score': float(overall_acc), 'confidence_interval_95': (float(lower_bound_acc), float(upper_bound_acc))},
                        'n': n}

        return overall_scores
    
    def __get_existing_results(self):
        if self.output_filepath and os.path.exists(self.output_filepath):
            with open(self.output_filepath, 'r') as fIn:
                existing_scores = json.load(fIn)
        else:
            existing_scores = {}
        return existing_scores  # conversation_id -> [{'utt_original': <str>, 'utt_noisy': <str>, 'tag_gold': <str>, 'tag_pred': <str>}]

    def __dump_existing_results(self, results_dict):
        if self.output_filepath:
            with open(self.output_filepath, 'w') as fOut:
                json.dump(results_dict, fOut, indent=4)

    def __get_reply_from_model(self, prompt):
        answer = self.model.generate(prompt)
        if self.debug:
            print(f'----------\n{prompt}\n---\n{answer}\n----------')
        return answer

    def __parse_reply(self, reply):
        label = ''
        explanation = ''
        reply_lines = reply.split('\n')
        for reply_line in reply_lines:
            reply_line = reply_line.lower().strip()
            if reply_line.startswith('label:'):
                label = reply_line[6:].strip()
            elif reply_line.startswith('explanation:'):
                explanation = reply_line[12:].strip()

        if 'statement' in label:
            label_to_use = 's'
        elif 'continuer' in label:
            label_to_use = 'b'
        elif 'floor holder' in label:
            label_to_use = 'fh'
        elif 'yes-no-question' in label:
            label_to_use = 'qy'
        elif 'interrupted' in label or 'abandoned' in label or 'uninterpretable' in label:
            label_to_use = '%'
        elif 'floor grabber' in label:
            label_to_use = 'fg'
        elif 'wh-question' in label or 'wh question' in label:
            label_to_use = 'qw'
        elif 'hold before answer' in label or 'agreement' in label:
            label_to_use = 'h'
        elif 'or-clause' in label or 'or clause' in label:
            label_to_use = 'qrr'
        elif 'rhetorical question' in label:
            label_to_use = 'qh'
        elif 'or question' in label:
            label_to_use = 'qr'
        elif 'open-ended question' in label or 'open ended question' in label:
            label_to_use = 'qo'
        else:
            label_to_use = ''

        return label_to_use, explanation

    
    def __classify_utterance(self, transcript_utterance):
        # prepare the prompt without the transcript part yet, to get the length:
        prompt =  f'Given an utterance from a conversation, choose a label that best describes the utterance.\n'
        prompt += f'The possible labels with their definitions are:\n'
        prompt += f'Floor Holder - the utterance occurs mid-speech and used by a speaker as a means to pause and continue holding the floor\n' # fh
        prompt += f'Floor Grabber - an utterance in which a speaker has not been speaking and wants to gain the floor so that he may commence speaking\n' # fg
        prompt += f'Hold Before Answer - an utterance that is used when a speaker who is given the floor and is expected to speak holds off prior to making an utterance\n' # h
        prompt += f'Agreement - an utterance used to exhibit agreement to or acceptance of a previous speaker\'s question, proposal, or statement\n' # h
        prompt += f'Yes-No-question - the utterance is in the form of a yes/no questions\n' # qy
        prompt += f'Wh-Question - the utterance is a question that require a specific answer\n' # qw
        prompt += f'Or-Clause - the utterance is an "or" clause, likely following a yes/no question\n' # qrr
        prompt += f'Or Question - the utterance offers the listener at least two answers or options from which to choose\n' # qr
        prompt += f'Open-ended Question - the utterance is an open-ended question that places few syntactic or semantic constraints on the form of the answer it elicits\n' # qo
        prompt += f'Rhetorical Question - the utterance states a question to which no answer is expected\n' # qh
        prompt += f'Abandoned/Interrupted - an incomplete utterance in which a speaker stops talking intentionally or on account of being interrupted by another speaker\n' # %
        prompt += f'Uninterpretable - the utterance is not clear or has indecipherable speech\n' # %
        prompt += f'Continuer - the utterance is made in the background and simply indicate that a listener is following along or at least is yielding the illusion that he is paying attention\n' # b (backchannel)
        prompt += f'Statement - the utterance is none of the above types\n\n' # s
        prompt += f"The utterance is:\n{transcript_utterance}\n\n"
        prompt += f'The output should be in the format:\n'
        prompt += f'label: <the label>\n'
        #prompt += f'explanation: <a short explanation for the choice>\n'

        # The manual explaining the labels are here: https://github.com/NathanDuran/MRDA-Corpus/blob/master/mrda_manual.pdf

        # send the prompt to the model and get an answer:
        reply = self.__get_reply_from_model(prompt)
        label, _ = self.__parse_reply(reply)

        return label

    def __get_utterance_indices_to_use(self, conversation_utterances_list):
        num_utts = len(conversation_utterances_list)
        if num_utts <= self.num_utterances_per_conversation:
            return list(range(num_utts))
        else:
            # use the first and last n/2 indices
            start_idxs = list(range(int(self.num_utterances_per_conversation / 2)))
            end_idxs = list(range(int(num_utts - (self.num_utterances_per_conversation / 2)), num_utts))
            return start_idxs + end_idxs


    def process_dataset(self):
        conversation_id_to_results = self.__get_existing_results()
        
        for conversation_id in tqdm(self.data):
            # get the indices of the utterances to process:
            utterance_indices_to_use = self.__get_utterance_indices_to_use(self.data[conversation_id])
            
            # get the utterance indices already processed:
            utterance_indices_already_processed = set()
            if conversation_id in conversation_id_to_results:
                for utt in conversation_id_to_results[conversation_id]:
                    utterance_indices_already_processed.add(utt['idx'])
            else:
                conversation_id_to_results[conversation_id] = []
            logging.info(f'On conversation {conversation_id}, already processed {len(utterance_indices_already_processed)} of {self.num_utterances_per_conversation} utterances.')

            #for utterance_idx, utterance_info in enumerate(self.data[conversation_id]):
            for utterance_idx in tqdm(utterance_indices_to_use):
                if utterance_idx in utterance_indices_already_processed:
                    continue

                utterance_info = self.data[conversation_id][utterance_idx]

                label_pred = self.__classify_utterance(utterance_info['text'])
                utterance_result = {
                    'idx': utterance_idx,
                    'tag_gold': utterance_info['tag_general'],
                    'tag_pred': label_pred,
                    'utt_noisy': utterance_info['text'],
                    'utt_original': utterance_info['text_original'],
                    'speaker': utterance_info['speaker']
                }
                conversation_id_to_results[conversation_id].append(utterance_result)

                if (not self.debug and (utterance_idx + 1) % 50 == 0) or (self.debug and (utterance_idx + 1) % 5 == 0):
                    logging.info(f'\tDumping results at index {utterance_idx} of {len(self.data[conversation_id])}')
                    self.__dump_existing_results(conversation_id_to_results)
                    if self.debug:
                        break

            self.__dump_existing_results(conversation_id_to_results)
            logging.info(f'\tFinished conversation {conversation_id}')
        
        # if not done so already, compute the overall scores:
        if 'overall' not in conversation_id_to_results:
            # get the precited and gold labels from all the utterances together:
            all_labels_pred = []
            all_labels_gold = []
            for conversation_id in conversation_id_to_results:
                for utt_result in conversation_id_to_results[conversation_id]:
                    all_labels_pred.append(utt_result['tag_pred'])
                    all_labels_gold.append(utt_result['tag_gold'])

            # compute the overall classification score:
            overall_scores = self.__compute_overall_scores(all_labels_gold, all_labels_pred)
            conversation_id_to_results['overall'] = overall_scores
            self.__dump_existing_results(conversation_id_to_results)
        
        return conversation_id_to_results['overall']


models_loaded = {'mistral': None, 'llama3': None, 'llama31': None, 'gpt': None}
def main(input_mrda_data_folderpath, output_folderpath, model_id, alternative_transcript_path=None, is_debug=False):
    # there might be an alternative transcript jsonl file to use:
    output_filename_suffix = ''
    if alternative_transcript_path is not None:
        if '_cleaned_' in alternative_transcript_path:
            output_filename_suffix_idx = alternative_transcript_path.index('_cleaned_')
            output_filename_suffix = alternative_transcript_path[output_filename_suffix_idx:-6]  # the cleaning suffix without '.jsonl'

    # set the processing method and output filename according to the specified configuration:
    if model_id == 'mistral':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelMistral7BInstruct()
        model = models_loaded[model_id]
        output_filename = f'results_Mistral7BInstruct{output_filename_suffix}.json'
    elif model_id == 'llama3':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelLlama3Instruct()
        model = models_loaded[model_id]
        output_filename = f'results_Llama3Instruct{output_filename_suffix}.json'
    elif model_id == 'llama31':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelLlama3Instruct()
        model = models_loaded[model_id]
        output_filename = f'results_Llama3_1Instruct{output_filename_suffix}.json'
    elif model_id == 'gpt':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelGPT4oMini()
        model = models_loaded[model_id]
        output_filename = f'results_Gpt4oMini{output_filename_suffix}.json'
        
    output_filepath = os.path.join(output_folderpath, output_filename)

    logging.basicConfig(level=logging.INFO, 
                        filename=f"logfile_inference_qaconv_{model_id}",
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    processor = ProcessMRDA(model=model, 
                            input_mrda_data_folderpath=input_mrda_data_folderpath, 
                            input_mrda_data_transcript_filepath=alternative_transcript_path,
                            output_filepath=output_filepath,
                            debug=is_debug)
    
    overall_scores = processor.process_dataset()
    logging.info(overall_scores)



if __name__ == '__main__':
    # usage:
    # python3 run_inference_qaconv.py <model_name>
    # model_name: one of mistral | llama3 | llama31 | gpt
    #
    # Variables to set:
    #   BASE_PATH: the base path for this project
    BASE_PATH = config.BASE_PATH

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'mistral'
    is_debug = config.DEBUG

    input_mrda_data_folderpath = f'{BASE_PATH}/data/MRDA/test'

    dirs_to_process = [
        (None, f'{BASE_PATH}/results/mrda_test_source'),
        (f'{BASE_PATH}/transcripts/mrda_test', f'{BASE_PATH}/results/mrda_test'),
        (f'{BASE_PATH}/transcripts/mrda_test___reverb__noise_10', f'{BASE_PATH}/results/mrda_test___reverb__noise_10'),
        (f'{BASE_PATH}/transcripts/mrda_test___reverb__noise_5', f'{BASE_PATH}/results/mrda_test___reverb__noise_5'),
        (f'{BASE_PATH}/transcripts/mrda_test___reverb__noise_0', f'{BASE_PATH}/results/mrda_test___reverb__noise_0'),
        (f'{BASE_PATH}/transcripts/mrda_test___reverb__noise_-5', f'{BASE_PATH}/results/mrda_test___reverb__noise_-5'),
        (f'{BASE_PATH}/transcripts/mrda_test___reverb__noise_-10', f'{BASE_PATH}/results/mrda_test___reverb__noise_-10')
    ]


    for input_dir_path, output_dir_path in dirs_to_process:

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        if input_dir_path is None:
            print(f'Processing: {input_mrda_data_folderpath}')
            main(input_mrda_data_folderpath, output_dir_path, model_name, alternative_transcript_path=None, is_debug=is_debug)
        else:
            for filename in os.listdir(input_dir_path):
                if filename.startswith('whisper_small__mrda_'):
                    transcription_path = os.path.join(input_dir_path, filename)
                    print(f'Processing: {transcription_path}')
                    main(input_mrda_data_folderpath, output_dir_path, model_name, alternative_transcript_path=transcription_path, is_debug=is_debug)