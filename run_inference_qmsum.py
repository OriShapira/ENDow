import sys
import evaluate
import os
import json
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import logging

import config
from generation_models import GenModelMistral7BInstruct, GenModelLlama3Instruct, GenModelLlama3_1Instruct, GenModelGPT4oMini


class ProcessQMSUM:
    METHOD_TRUNCATE = 1
    METHOD_RECURSIVE = 2
    METHOD_FULL = 3

    def __init__(self, model, processing_method,
                 input_qmsum_data_folderpath, 
                 input_qmsum_data_transcript_filepath=None,
                 output_filepath=None, debug=False):
        '''
        model_class: a class based on GenModelBase which will be used to summarize QMSum
        processing_method: ProcessQMSUM.METHOD_TRUNCATE or ProcessQMSUM.METHOD_RECURSIVE
        input_qmsum_data_folderpath: the path to the QMSum data split directory
        input_qmsum_data_transcript_filepath: if there is an alternative QMSum transcript file (e.g., noised)
            set the path to the file. This will be used for summarizing and scoring instead of the original
            QMSum transcript from input_qmsum_data_folderpath. Otherwise set this with None.
        output_filepath: where to output the summaries and scores
        '''
        self.debug = debug
        if self.debug:
            print('DEBUG MODE')
        self.data_path = input_qmsum_data_folderpath
        self.data_transcript_filepath = input_qmsum_data_transcript_filepath
        self.output_filepath = output_filepath
        if processing_method == ProcessQMSUM.METHOD_TRUNCATE:
            self.processing_func = self.__summarize_truncate
        elif processing_method == ProcessQMSUM.METHOD_RECURSIVE:
            self.processing_func = self.__summarize_recursive
        elif processing_method == ProcessQMSUM.METHOD_FULL:
            self.processing_func = self.__summarize_full
        else:
            raise Exception('The method requested for processing is not implemented.')
        self.rouge_metric = evaluate.load('rouge')
        self.model = model
        self.data = self.__load_data()
        
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
        return conversation_id_to_utterances

    def __load_data(self):
        alternative_transcripts = self.__load_alternative_transcript_file()

        data = {}  # datum_name -> {'transcript': <str>, 'queries': ['query': <str>, 'type': <'generic'|'query'>, 'reference': <str>, 'relevant_section': <str>}]}
        for filename in os.listdir(self.data_path):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(self.data_path, filename)
            conversation_id = filename[:-5]
            transcript_parts = []
            queries = []
            with open(filepath, 'r') as fIn:
                datum = json.load(fIn)
                # get the transcript utterances:
                for utt_idx, utt_info in enumerate(datum['meeting_transcripts']):
                    # if there is an alternative trancript to use, use the utterance from there:
                    if conversation_id in alternative_transcripts:
                        if utt_idx in alternative_transcripts[conversation_id]:
                            utt = alternative_transcripts[conversation_id][utt_idx]
                        else:
                            utt = ''
                    else:  # otherwise use the utterance from the original QMSum data
                        utt = utt_info["content"]
                    # append the utterance to the transcript:
                    transcript_parts.append(f'{utt_info["speaker"]}: {utt}')
                # get the generic summaries information
                for query_info in datum['general_query_list']:
                    query = query_info['query']
                    answer_gold = query_info['answer']
                    queries.append({'query': query, 'type': 'generic', 'reference': answer_gold})
                # get the queries information
                for query_info in datum['specific_query_list']:
                    query = query_info['query']
                    answer_gold = query_info['answer']
                    span_idxs_list = query_info['relevant_text_span']
                    #relevant_sections = ''
                    #for span_idxs in span_idxs_list:
                    #    relevant_sections += ' '.join([f'{datum["meeting_transcripts"][utt_idx]["speaker"]}: {datum["meeting_transcripts"][utt_idx]["content"]}'
                    #                                  for utt_idx in range(int(span_idxs[0]), int(span_idxs[1]))])
                    queries.append({'query': query, 'type': 'query', 'reference': answer_gold, 'relevant_sections': span_idxs_list})
            #data[filename] = {'transcript': ' '.join(transcript_parts), 'queries': queries}
            data[filename] = {'transcript': transcript_parts, 'queries': queries}
        return data

    def __compute_scores(self, summ_predictions, summ_references):
        results = self.rouge_metric.compute(predictions=summ_predictions, references=summ_references, 
                                            rouge_types=['rouge1', 'rouge2', 'rougeL'], use_aggregator=False)
        return results  # {'rouge1': [<score per instance>], 'rouge2': [<score per instance>], 'rougeL': [<score per instance>]}
    
    def __compute_confidence_interval(self, values, confidence_level=0.95):
        # Computes the confidence interval according to the confidence_level specified.
        # E.g., for confidence_level=0.95, the confidence interval is for the (0.025, 0.975) percentiles.
        # The assumption is that the list of values converges to a normal distibution.
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard Error of the Mean
        n = len(values)
        critical_value = stats.t.ppf((1 + confidence_level) / 2., n - 1)  # t-distribution
        margin_of_error = critical_value * sem
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)
        return confidence_interval

    def __compute_overall_scores(self, scores_dict):
        # scores_dict is expected to be a dictionary with metric names as keys and lists of scores as values
        overall_scores = {}
        for metric in scores_dict:
            overall_scores[metric] = {
                'mean': np.mean(scores_dict[metric]),
                'std': np.std(scores_dict[metric]),
                'median': np.median(scores_dict[metric]),
                'quartile_1': np.percentile(scores_dict[metric], 25),
                'quartile_3': np.percentile(scores_dict[metric], 75),
                'confidence_interval_95': self.__compute_confidence_interval(scores_dict[metric]),
                'n': len(scores_dict[metric])
            }
        return overall_scores
    
    def __get_existing_scores(self):
        if self.output_filepath and os.path.exists(self.output_filepath):
            with open(self.output_filepath, 'r') as fIn:
                existing_scores = json.load(fIn)
        else:
            existing_scores = {}
        return existing_scores  # datum_name -> <generic|query_focused> -> metric -> list of scores

    def __dump_existing_scores(self, scores_dict):
        if self.output_filepath:
            with open(self.output_filepath, 'w') as fOut:
                json.dump(scores_dict, fOut, indent=4)

    def __get_transcript_chunk(self, transcript_utterances, start_idx, max_len):
        chunk_str = ''
        chunk_token_len = 0
        last_utt_idx = -1
        for utt_idx_relative, utt in enumerate(transcript_utterances[start_idx:]):
            utt_token_len = len(utt.split())
            if chunk_token_len + utt_token_len >= max_len:
                break  # don't add this chunk because it will cause surpassing of the max_len
            chunk_str += ' ' + utt
            chunk_token_len += utt_token_len
            last_utt_idx = start_idx + utt_idx_relative
        return chunk_str, last_utt_idx

    def __get_reply_from_model(self, prompt):
        answer = self.model.generate(prompt)
        if self.debug:
            print(f'----------\n{prompt}\n---\n{answer}\n----------')
        return answer
    
    def __summarize_truncate(self, transcript_utterances, query):
        # prepare the prompt without the transcript part yet, to get the length:
        prompt =  f"Given the following conversation, answer the question: {query}\n"
        prompt += f"The conversation is:\n[TRANSCRIPT]"
        prompt_token_len = len(prompt.split())
        
        # get the truncated transcript, upto the maximum conext length of the model:
        input_str, _ = self.__get_transcript_chunk(transcript_utterances, 0, self.model.max_context_len() - prompt_token_len)
        
        # add the transcript part to the prompt:
        prompt = prompt.replace('[TRANSCRIPT]', input_str)

        # send the prompt to the model and get an answer:
        answer = self.__get_reply_from_model(prompt)

        return answer

    def __summarize_recursive(self, transcript_utterances, query):
        # loop chunk by chunk of the transcript:
        last_utterance_idx = -1
        answers_so_far = []
        while last_utterance_idx != len(transcript_utterances) - 1:
            # prepare the prompt without the transcript part yet, to get the length:
            prompt =  f"Given the following portion of a conversation, answer the question: {query}\n"
            prompt += f"The portion of the conversation is:\n[TRANSCRIPT]"
            
            prompt_token_len = len(prompt.split())
            
            # get the truncated transcript, upto the maximum conext length of the model:
            input_str, last_utterance_idx = self.__get_transcript_chunk(transcript_utterances, 
                                                                        last_utterance_idx + 1,
                                                                        self.model.max_context_len() - prompt_token_len)
            
            # add the transcript part to the prompt:
            prompt = prompt.replace('[TRANSCRIPT]', input_str)

            # send the prompt to the model and get an answer:
            answer = self.__get_reply_from_model(prompt)
            answers_so_far.append(answer)
        
        # now get a summary of summaries:
        prompt =  'The following is an ordered list of answers collected from portions of a conversation '
        prompt += f'for the question: {query}\n'
        prompt += 'Generate a final answer for the question by aggregating the answers from the '
        prompt += 'different conversation portions. Be succinct, and write it as a standalone answer '
        prompt += 'without referring to the list of existing answers.'
        prompt += 'The answers are: \n'
        prompt += '\n'.join([f'Answer {i + 1}: {answers_so_far[i]}' for i in range(len(answers_so_far))]) + '\n'

        # send the prompt to the model and get an answer:
        final_answer = self.__get_reply_from_model(prompt)

        return final_answer
    
    def __summarize_full(self, transcript_utterances, query):
        # prepare the prompt without the transcript part yet, to get the length:
        prompt =  f"Given the following conversation, answer the question: {query}\n"
        prompt += f"The conversation is:\n[TRANSCRIPT]"
        prompt_token_len = len(prompt.split())
        
        # get the truncated transcript, upto the maximum context length of the model:
        input_str, _ = self.__get_transcript_chunk(transcript_utterances, 0, 999999999)
        
        # add the transcript part to the prompt:
        prompt = prompt.replace('[TRANSCRIPT]', input_str)

        # send the prompt to the model and get an answer:
        answer = self.__get_reply_from_model(prompt)

        return answer
    
    def process_dataset(self):
        datum_to_scores = self.__get_existing_scores()
        
        for datum_name in tqdm(self.data):
            logging.info(f'On datum {datum_name}')
            if datum_name in datum_to_scores:
                logging.info(f'\tSkipping because already processed: {datum_name}')
                continue  # this datum has already been computed

            summ_references_gen = []
            summ_predictions_gen = []
            summ_references_que = []
            summ_predictions_que = []
            for query_idx, query_info in tqdm(enumerate(self.data[datum_name]['queries'])):
                logging.info(f'\tOn query {query_idx} of {len(self.data[datum_name]["queries"])}')
                reply = self.processing_func(self.data[datum_name]['transcript'], query_info['query'])
                if query_info['type'] == 'generic':
                    summ_references_gen.append(query_info['reference'])
                    summ_predictions_gen.append(reply)
                elif query_info['type'] == 'query':
                    summ_references_que.append(query_info['reference'])
                    summ_predictions_que.append(reply)
            datum_scores_gen = self.__compute_scores(summ_predictions_gen, summ_references_gen)
            datum_scores_que = self.__compute_scores(summ_predictions_que, summ_references_que)
            
            # keep the computed scores in a file:
            datum_to_scores[datum_name] = {'generic': datum_scores_gen, 
                                           'query_focused': datum_scores_que, 
                                           'generic_summaries': {'references': summ_references_gen, 'predictions': summ_predictions_gen},
                                           'query_focused_summaries': {'references': summ_references_que, 'predictions': summ_predictions_que}}
            self.__dump_existing_scores(datum_to_scores)

            if self.debug:
                break
        
        # if not done so already, compute the overall scores:
        if 'overall' not in datum_to_scores:
            # aggregate the scores from all datums
            all_scores_generic = {}  # metric -> list of scores from all datums
            all_scores_query_focused = {}  # metric -> list of scores from all datums
            for datum_name in datum_to_scores:
                for metric in datum_to_scores[datum_name]['generic']:
                    if metric not in all_scores_generic:
                        all_scores_generic[metric] = []
                        all_scores_query_focused[metric] = []
                    all_scores_generic[metric].extend(datum_to_scores[datum_name]['generic'][metric])
                    all_scores_query_focused[metric].extend(datum_to_scores[datum_name]['query_focused'][metric])
            all_scores = {metric: all_scores_generic[metric] + all_scores_query_focused[metric] for metric in all_scores_generic}

            # compute the scores over all data:
            overall_scores_generic = self.__compute_overall_scores(all_scores_generic)
            overall_scores_query_focused = self.__compute_overall_scores(all_scores_query_focused)
            overall_scores = self.__compute_overall_scores(all_scores)
            datum_to_scores['overall'] = {
                'generic': overall_scores_generic,
                'query_focused': overall_scores_query_focused,
                'all': overall_scores,
            }
            self.__dump_existing_scores(datum_to_scores)
        
        return datum_to_scores['overall']



models_loaded = {'mistral': None, 'llama3': None, 'llama31': None, 'gpt': None}
def main(input_qmsum_data_folderpath, output_folderpath, model_id, method_name, alternative_transcript_path=None, is_debug=False):
    # there might be an alternative transcript jsonl file to use:
    output_filename_suffix = ''
    if alternative_transcript_path is not None:
        #output_folderpath = os.path.dirname(alternative_transcript_path)
        if '_cleaned_' in alternative_transcript_path:
            output_filename_suffix_idx = alternative_transcript_path.index('_cleaned_')
            output_filename_suffix = alternative_transcript_path[output_filename_suffix_idx:-6]  # the cleaning suffix without '.jsonl'
    
    # set the processing method and output filename according to the specified configuration:
    if model_id == 'mistral':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelMistral7BInstruct()
        model = models_loaded[model_id]
        if method_name == 'recursive':
            processing_method = ProcessQMSUM.METHOD_RECURSIVE
            output_filename = f'results_Mistral7BInstruct_recursive{output_filename_suffix}.json'
        elif method_name == 'truncate':
            processing_method = ProcessQMSUM.METHOD_TRUNCATE
            output_filename = f'results_Mistral7BInstruct_truncate{output_filename_suffix}.json'
    elif model_id == 'llama3':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelLlama3Instruct()
        model = models_loaded[model_id]
        if method_name == 'recursive':
            processing_method = ProcessQMSUM.METHOD_RECURSIVE
            output_filename = f'results_Llama3Instruct_recursive{output_filename_suffix}.json'
        elif method_name == 'truncate':
            processing_method = ProcessQMSUM.METHOD_TRUNCATE
            output_filename = f'results_Llama3Instruct_truncate{output_filename_suffix}.json'
    elif model_id == 'llama31':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelLlama3_1Instruct()
        model = models_loaded[model_id]
        if method_name == 'recursive':
            processing_method = ProcessQMSUM.METHOD_RECURSIVE
            output_filename = f'results_Llama3_1Instruct_recursive{output_filename_suffix}.json'
        elif method_name == 'truncate':
            processing_method = ProcessQMSUM.METHOD_TRUNCATE
            output_filename = f'results_Llama3_1Instruct_truncate{output_filename_suffix}.json'
    elif model_id == 'gpt':
        if models_loaded[model_id] is None:
            models_loaded[model_id] = GenModelGPT4oMini()
        model = models_loaded[model_id]
        if method_name == 'recursive':
            processing_method = ProcessQMSUM.METHOD_RECURSIVE
            output_filename = f'results_Gpt4oMini_recursive{output_filename_suffix}.json'
        elif method_name == 'truncate':
            processing_method = ProcessQMSUM.METHOD_TRUNCATE
            output_filename = f'results_Gpt4oMini_truncate{output_filename_suffix}.json'
        
    output_filepath = os.path.join(output_folderpath, output_filename)

    logging.basicConfig(level=logging.INFO, 
                        filename=f"logfile_inference_qmsum_{model_id}_{method_name}",
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    processor = ProcessQMSUM(model=model, 
                             processing_method=processing_method,
                             input_qmsum_data_folderpath=input_qmsum_data_folderpath, 
                             input_qmsum_data_transcript_filepath=alternative_transcript_path,
                             output_filepath=output_filepath,
                             debug=is_debug)
    
    overall_scores = processor.process_dataset()
    logging.info(overall_scores)



if __name__ == '__main__':
    # usage:
    # python3 run_inference_qmsum.py <model_name> <method_name> [optional: --debug]
    # model_name: one of mistral | llama3 | llama31 | gpt
    # method_name: one of recursive | truncate (for summarization method)
    #
    # Variables to set:
    #   BASE_PATH: the base path for this project
    BASE_PATH = config.BASE_PATH

    if len(sys.argv) >= 3:
        model_name = sys.argv[1]
        method_name = sys.argv[2]
        is_debug = True if '--debug' in sys.argv else False
    else:
        model_name = 'mistral'
        method_name = 'recursive'
        is_debug = True
        print('Running in debug mode.')

    input_qmsum_data_folderpath = f'{BASE_PATH}/data/QMSum/test'
    dirs_to_process = [
        (None, f'{BASE_PATH}/results/qmsum_test_source'),
        (f'{BASE_PATH}/transcripts/qmsum_test', f'{BASE_PATH}/results/qmsum_test'),
        (f'{BASE_PATH}/transcripts/qmsum_test___reverb__noise_10', f'{BASE_PATH}/results/qmsum_test___reverb__noise_10'),
        (f'{BASE_PATH}/transcripts/qmsum_test___reverb__noise_5', f'{BASE_PATH}/results/qmsum_test___reverb__noise_5'),
        (f'{BASE_PATH}/transcripts/qmsum_test___reverb__noise_0', f'{BASE_PATH}/results/qmsum_test___reverb__noise_0'),
        (f'{BASE_PATH}/transcripts/qmsum_test___reverb__noise_-5', f'{BASE_PATH}/results/qmsum_test___reverb__noise_-5'),
        (f'{BASE_PATH}/transcripts/qmsum_test___reverb__noise_-10', f'{BASE_PATH}/results/qmsum_test___reverb__noise_-10')
    ]

    for input_dir_path, output_dir_path in dirs_to_process:

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        if input_dir_path is None:
            print(f'Processing: {input_qmsum_data_folderpath}')
            main(input_qmsum_data_folderpath, output_dir_path, model_name, method_name, alternative_transcript_path=None, is_debug=is_debug)
        else:
            for filename in os.listdir(input_dir_path):
                if filename.startswith('whisper_small__qmsum_'):
                    transcription_path = os.path.join(input_dir_path, filename)
                    print(f'Processing: {transcription_path}')
                    main(input_qmsum_data_folderpath, output_dir_path, model_name, method_name, alternative_transcript_path=transcription_path, is_debug=is_debug)