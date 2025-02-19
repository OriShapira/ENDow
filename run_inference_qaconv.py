import sys
import os
import json
from tqdm import tqdm
import logging
import re
import traceback

import config
from generation_models import GenModelMistral7BInstruct, GenModelLlama3Instruct, GenModelLlama3_1Instruct, GenModelGPT4oMini
from qaconv_eval import evaluate_answers as qa_conv_eval
from qaconv_eval import make_eval_dicts as qa_conv_eval_overall




class ProcessQAConv:

    def __init__(self, model,
                 input_qaconv_data_qa_filepath, 
                 input_qaconv_data_transcript_source_filepath,
                 input_qaconv_data_transcript_alternative_filepath=None,
                 output_filepath=None, debug=False):
        '''
        model_class: a class based on GenModelBase which will be used to solve QA for QAConv
        input_qaconv_data_qa_filepath: the path to the QAConv question-answers data
        input_qaconv_data_transcript_source_filepath: the path to the QAConv source transcripts data (article_full.json)
        input_qaconv_data_transcript_alternative_filepath: an optional transcript file that was noised/cleaned, which is in a different jsonl format (None if using the source transcript)
        output_filepath: where to output the answers and scores
        '''
        self.debug = debug
        if self.debug:
            print('DEBUG MODE')
        self.data_qa_path = input_qaconv_data_qa_filepath
        self.data_transcripts_source_path = input_qaconv_data_transcript_source_filepath
        self.data_transcripts_alternative_path = input_qaconv_data_transcript_alternative_filepath
        self.output_filepath = output_filepath
        self.model = model
        self.data = self.__load_data()  # conversation_id -> {'transcript': [<str>], 'qa': [{'id': <str>, 'question': <str>, 'answers': [<str>]}]}
        
    def __load_qa_from_file(self):
        conversation_id_to_qa = {}
        with open(self.data_qa_path, 'r') as fIn:
            qa_file_data = json.load(fIn)
            for qa_instance in qa_file_data:
                conversation_id = qa_instance['article_full_id'][0]
                if 'court' in conversation_id or 'newsidal' in conversation_id:
                    qa_instance_id = qa_instance['id']
                    qa_question = qa_instance['question']
                    qa_answers = qa_instance['answers']
                    if conversation_id not in conversation_id_to_qa:
                        conversation_id_to_qa[conversation_id] = []
                    conversation_id_to_qa[conversation_id].append({'id': qa_instance_id, 'question': qa_question, 'answers': qa_answers})
        return conversation_id_to_qa  # conversation_id -> [{'id': <str>, 'question': <str>, 'answers': [<str>]}]
    
    def __load_transcripts_from_source_file(self, conversation_ids_to_use):
        conversation_id_to_utterances = {}
        with open(self.data_transcripts_source_path, 'r') as fIn:
            data = json.load(fIn)
            for conversation_id in data:
                if conversation_id not in conversation_ids_to_use:
                    continue
                for utt_info in data[conversation_id]:
                    utt_text = utt_info['text']
                    utt_id = utt_info['id']
                    utt_speaker = utt_info['speaker']
                    if conversation_id not in conversation_id_to_utterances:
                        conversation_id_to_utterances[conversation_id] = []
                    conversation_id_to_utterances[conversation_id].append({'id': utt_id, 'speaker': utt_speaker, 'text': utt_text})
        return conversation_id_to_utterances  # conversation_id -> [{'id': <str>, 'speaker': <str>, 'text': <str>}]

    def __update_transcripts_from_alternative_file(self, conversation_id_to_utterances_source):
        if self.data_transcripts_alternative_path is not None:
            # convert the source utterances from to:
            #   conversation_id -> [{'id': <str>, 'speaker': <str>, 'text': <str>}]
            #   conversation_id -> utterance_id -> {'id': <str>, 'speaker': <str>, 'text': <str>}
            conversation_id_to_utt_id_to_utterance_info_source = {conversation_id: {utt_info['id']: utt_info 
                                                                                    for utt_info in conversation_id_to_utterances_source[conversation_id]}
                                                                  for conversation_id in conversation_id_to_utterances_source}
            # get the text of the alternative utterances, but use the speaker of each utterance from the source data:
            conversation_id_to_utterances = {}
            with open(self.data_transcripts_alternative_path, 'r') as fIn:
                for line in fIn:
                    datum = json.loads(line.strip())
                    utt_id = datum['id']
                    conversation_id = '-'.join(utt_id.split('-')[:-1])
                    #utt_idx = int(utt_id.split('_')[-1])
                    if conversation_id not in conversation_id_to_utt_id_to_utterance_info_source:
                        continue
                    utt_text = datum['pred']  # the alternative text for the utterance
                    if conversation_id not in conversation_id_to_utterances:
                        conversation_id_to_utterances[conversation_id] = []
                    utt_speaker = conversation_id_to_utt_id_to_utterance_info_source[conversation_id][utt_id]['speaker']
                    conversation_id_to_utterances[conversation_id].append(f'{utt_speaker}: {utt_text}')
        else:
            # if there's no alternative transcript, just convert the source utterances to "<speaker>: <txt>" format:
            conversation_id_to_utterances = {conversation_id: [f'{utt_info["speaker"]}: {utt_info["text"]}'
                                                               for utt_info in conversation_id_to_utterances_source[conversation_id]]
                                             for conversation_id in conversation_id_to_utterances_source}
        
        return conversation_id_to_utterances  # conversation_id -> [utterances_with_speaker]


    def __load_data(self):
        # get the QAs and transcripts:
        conversation_id_to_qa = self.__load_qa_from_file()  # conversation_id -> [{'id': <str>, 'question': <str>, 'answers': [<str>]}]
        conversation_ids_to_use = list(conversation_id_to_qa.keys())
        conversation_id_to_utterances_source = self.__load_transcripts_from_source_file(conversation_ids_to_use)
        conversation_id_to_utterances = self.__update_transcripts_from_alternative_file(conversation_id_to_utterances_source)  # conversation_id -> [utterances_with_speaker]
        
        # combine to a single dictionary
        data = {}  # conversation_id -> {'transcript': [<str>], 'qa': [{'id': <str>, 'question': <str>, 'answers': [<str>]}]}
        for conversation_id in conversation_id_to_qa:
            if conversation_id in conversation_id_to_utterances:
                data[conversation_id] = {'transcript': conversation_id_to_utterances[conversation_id],
                                         'qa': conversation_id_to_qa[conversation_id]}
        return data

    def __compute_scores(self, answers_predictions, answers_references):
        '''
        answers_predictions: a list of answers as predicted by a model
        answers_references: a list, ordered in the same question-order as answers_predictions, where each item in the list is a list of possible answers. Usually there's one answer but sometimes there can be several.
        '''
        overall_scores, all_scores = qa_conv_eval(answers_references, answers_predictions)
        # overall_scores {exact: {<stats>}, f1: {<stats>}, fzr: {<stats>}, unans_f1: <float>}
        # all_scores {'exact': [<score per instance>], 'f1': [<score per instance>], 'fzr': [<score per instance>], 'unans_gold': [<value per instance>], 'unans_pred': [<value per instance>]}
        return all_scores

    def __compute_overall_scores(self, scores_dict):
        overall_scores, _ = qa_conv_eval_overall(scores_dict['exact'], scores_dict['f1'], scores_dict['fzr'], None,  scores_dict['unans_gold'],  scores_dict['unans_pred'])
        return overall_scores
    
    def __get_existing_scores(self):
        if self.output_filepath and os.path.exists(self.output_filepath):
            with open(self.output_filepath, 'r') as fIn:
                existing_scores = json.load(fIn)
        else:
            existing_scores = {}
        return existing_scores  # datum_name -> metric -> list of scores

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
            chunk_str += '\n' + utt
            chunk_token_len += utt_token_len
            last_utt_idx = start_idx + utt_idx_relative
        return chunk_str, last_utt_idx

    def __get_reply_from_model(self, prompt):
        reply = self.model.generate(prompt)
        if self.debug:
            print(f'----------\n{prompt}\n---\n{reply}\n----------')
        return reply

    def __parse_qa_reply(self, reply, num_questions):
        
        # Split the string by lines
        lines = reply.split("\n")

        if len(lines) < num_questions:
            raise ValueError(f"Not enough lines in the reply. Need {num_questions} but have {len(lines)}: {reply}")
        elif len(lines) > num_questions:
            # if there are more lines than questions, there might be an opening statement, and in this case, there is likely numbering for each question,
            # so we move forward in the lines until we reach a line that starts with "1.":
            line_idx_to_start = 0
            while line_idx_to_start < len(lines) and not lines[line_idx_to_start].startswith('1.'):
                line_idx_to_start += 1
            lines = lines[line_idx_to_start:]
            lines = [l for l in lines if l.strip() != '']  # remove empty lines
            if len(lines) != num_questions:
                raise ValueError(f"Wrong number of lines in the reply. Need {num_questions} but have {len(lines)}: {reply}")

        # loop over the lines in the reply and remove the numbering, in case there is, to get the answers:
        answers = []
        for line in lines:
            # If the line starts with  "<num>. <text>", just get the text:
            if re.match(r"^\d+\.\s.+", line):
                # Extract and add the text after the line number
                line_txt = line.split(".", 1)[1].strip()
            elif line.lower().startswith('answer:'):
                line_txt = line.split(":", 1)[1].strip()
            else:
                line_txt = line.strip()
            
            # remove anything within parentheses in the text in case there is anything like that:
            line_txt = re.sub(r"\s*\(.*?\)\s*", " ", line_txt).strip()
            # if the text includes the "unanswerable" string, make sure it has nothing else:
            if "unanswerable" in line_txt.lower():
                line_txt = "unanswerable"
            answers.append(line_txt)

        return answers

    def __get_final_answers_from_chunks(self, answer_lists, num_questions):
        # answer_lists is a list of lists of answers. each list is the list of answers for the questions for a certain chunk.
        # This function sets the shortest answer that is not "unanswerable" as the answer for the question at that index.
        
        # if there are no answers, return empty strings for each of the questions:
        if len(answer_lists) == 0:
            return [""] * num_questions
        
        # otherwise use the shortest answer available for each question (or "unanswerable"):
        final_answers = ["unanswerable"] * num_questions
        for answers_list in answer_lists:
            for ans_idx, ans in enumerate(answers_list):
                # if the current answer for this question is "unanswerable", then use the first avaialble answer:
                if final_answers[ans_idx] == "unanswerable":
                    final_answers[ans_idx] = ans
                # otherwise use the shortest answer so far for this question:
                elif len(ans) <= len(final_answers[ans_idx]):
                    final_answers[ans_idx] = ans
        return final_answers
    
    def __answer_questions(self, transcript_utterances, questions, conversation_id):
        # loop chunk by chunk of the transcript:
        last_utterance_idx = -1
        answers_so_far = []
        while last_utterance_idx != len(transcript_utterances) - 1:
            # prepare the prompt without the transcript part yet, to get the length:
            prompt =  f"You will be given a conversation and some questions, and you need to answer the questions based on the conversation.\n"
            prompt += f"Each answer should be a very short span copied from the conversation, and written as a brief direct answer, and not as a sentence. Do not add any explanation or extra wording.\n"
            prompt += f"For example, for a question such as \"Where is John from?\", the answer could be \"New York\" but not \"John is from New York\".\n"
            prompt += f"If a question cannot be answered according to the conversation, answer with \"unanswerable\" only, without any explanation or extra wording.\n"
            prompt += f"Answer the questions line by line in the same order as the questions, without repeating the questions.\n\n"
            prompt += f"The conversation is:\n[TRANSCRIPT]\n\n"
            prompt += f"The questions are:\n[QUESTIONS]"
            
            prompt_token_len = len(prompt.split())
            
            # get the truncated transcript, upto the maximum conext length of the model:
            input_transcript_str, last_utterance_idx = self.__get_transcript_chunk(transcript_utterances, 
                                                                                   last_utterance_idx + 1,
                                                                                   self.model.max_context_len() - prompt_token_len)

            input_questions_str = '\n'.join([f'{question_idx + 1}. {question_txt}' for question_idx, question_txt in enumerate(questions)])
            
            # add the transcript part to the prompt:
            prompt = prompt.replace('[TRANSCRIPT]', input_transcript_str)
            prompt = prompt.replace('[QUESTIONS]', input_questions_str)

            # send the prompt to the model and get an answer:
            num_attempts = 1
            while num_attempts <= 3:
                try:
                    reply = self.__get_reply_from_model(prompt)
                    answers = self.__parse_qa_reply(reply, len(questions))
                    answers_so_far.append(answers)
                    break
                except Exception as e:
                    logging.info(e)
                    logging.error("Full traceback:\n%s", traceback.format_exc())
                    logging.info(f'Failed attempt number {num_attempts}')
                    traceback.print_exc()
                    print(f'Failed attempt number {num_attempts} in conversation {conversation_id}')
                    num_attempts += 1
        
        # get the final answer for each question, from all the chunks' answers:
        final_answers = self.__get_final_answers_from_chunks(answers_so_far, len(questions))

        return final_answers
    
    
    def process_dataset(self):
        conversation_id_to_scores = self.__get_existing_scores()

        # for each of the conversations, get answers for the questions and compute evalution scores:
        for conversation_id in tqdm(self.data):
            logging.info(f'On conversation {conversation_id}')
            
            
            # skip conversations that were already processed:
            if conversation_id in conversation_id_to_scores:
                logging.info(f'\tSkipping because already processed: {conversation_id}')
                continue  # this datum has already been computed

            # get answers for each of the questions of the current conversation:
            questions = [qa_info['question'] for qa_info in self.data[conversation_id]['qa']]
            answers_ref = [qa_info['answers'] for qa_info in self.data[conversation_id]['qa']]
            answers_pred = self.__answer_questions(self.data[conversation_id]['transcript'], questions, conversation_id)
            
            conversation_scores = self.__compute_scores(answers_pred, answers_ref)
                
            # keep the computed scores in a file:
            conversation_id_to_scores[conversation_id] = {'scores': conversation_scores, 
                                                          'answers': {'references': answers_ref, 'predictions': answers_pred}}
            self.__dump_existing_scores(conversation_id_to_scores)

            if self.debug:
                break
        
        # if not done so already, compute the overall scores:
        if 'overall' not in conversation_id_to_scores:
            # aggregate the scores from all conversations
            all_scores = {}  # metric -> list of scores from all conversations
            for conversation_id in conversation_id_to_scores:
                for metric in conversation_id_to_scores[conversation_id]['scores']:
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].extend(conversation_id_to_scores[conversation_id]['scores'][metric])

            # compute the scores over all data:
            overall_scores = self.__compute_overall_scores(all_scores)
            conversation_id_to_scores['overall'] = overall_scores
            self.__dump_existing_scores(conversation_id_to_scores)
        
        return conversation_id_to_scores['overall']



models_loaded = {'mistral': None, 'llama3': None, 'llama31': None, 'gpt': None}
def main(qaconv_data_qa_filepath, qaconv_data_transcript_source_filepath, output_folderpath, model_id,
         qaconv_data_transcript_alternative_filepath=None, is_debug=False):
    # there might be an alternative transcript jsonl file to use:
    output_filename_suffix = ''
    if qaconv_data_transcript_alternative_filepath is not None:
        if '_cleaned_' in qaconv_data_transcript_alternative_filepath:
            output_filename_suffix_idx = qaconv_data_transcript_alternative_filepath.index('_cleaned_')
            output_filename_suffix = qaconv_data_transcript_alternative_filepath[output_filename_suffix_idx:-6]  # the cleaning suffix without '.jsonl'

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
            models_loaded[model_id] = GenModelLlama3_1Instruct()
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

    processor = ProcessQAConv(model=model, 
                              input_qaconv_data_qa_filepath=qaconv_data_qa_filepath, 
                              input_qaconv_data_transcript_source_filepath=qaconv_data_transcript_source_filepath,
                              input_qaconv_data_transcript_alternative_filepath=qaconv_data_transcript_alternative_filepath,
                              output_filepath=output_filepath,
                              debug=is_debug)
    
    overall_scores = processor.process_dataset()
    logging.info(overall_scores)


if __name__ == '__main__':
    # usage:
    # python3 run_inference_qaconv.py <model_name> [optional: --debug]
    # model_name: one of mistral | llama3 | llama31 | gpt
    #
    # Variables to set:
    #   BASE_PATH: the base path for this project
    BASE_PATH = config.BASE_PATH

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        is_debug = True if '--debug' in sys.argv else False
    else:
        model_name = 'llama31' #'mistral' #'llama3'
        is_debug = True
        print('Running in debug mode.')


    input_qaconv_data_qa_filepath = f'{BASE_PATH}/data/QAConv/tst.json'
    input_qaconv_data_transcript_source_filepath = f'{BASE_PATH}/data/QAConv/article_full.json'
    dirs_to_process = [
        (None, f'{BASE_PATH}/results/qaconv_test_source'),
        (f'{BASE_PATH}/transcripts/qaconv_test', f'{BASE_PATH}/results/qaconv_test'),
        (f'{BASE_PATH}/transcripts/qaconv_test___reverb__noise_10', f'{BASE_PATH}/results/qaconv_test___reverb__noise_10'),
        (f'{BASE_PATH}/transcripts/qaconv_test___reverb__noise_5', f'{BASE_PATH}/results/qaconv_test___reverb__noise_5'),
        (f'{BASE_PATH}/transcripts/qaconv_test___reverb__noise_0', f'{BASE_PATH}/results/qaconv_test___reverb__noise_0'),
        (f'{BASE_PATH}/transcripts/qaconv_test___reverb__noise_-5', f'{BASE_PATH}/results/qaconv_test___reverb__noise_-5'),
        (f'{BASE_PATH}/transcripts/qaconv_test___reverb__noise_-10', f'{BASE_PATH}/results/qaconv_test___reverb__noise_-10')
    ]

    for input_dir_path, output_dir_path in dirs_to_process:

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        if input_dir_path is None:
            print(f'Processing: {input_qaconv_data_transcript_source_filepath}')
            main(input_qaconv_data_qa_filepath, input_qaconv_data_transcript_source_filepath, output_dir_path, model_name,
                 qaconv_data_transcript_alternative_filepath=None, is_debug=is_debug)
        else:
            for filename in os.listdir(input_dir_path):
                if filename.startswith('whisper_small__qaconv_'): # and '_cleaned_' in filename:
                    transcription_path = os.path.join(input_dir_path, filename)
                    print(f'Processing: {transcription_path}')
                    main(input_qaconv_data_qa_filepath, input_qaconv_data_transcript_source_filepath, output_dir_path, model_name,
                        qaconv_data_transcript_alternative_filepath=transcription_path, is_debug=is_debug)