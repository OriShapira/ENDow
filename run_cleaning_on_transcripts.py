import spacy
import jiwer
from collections import Counter
from whisper_normalizer.english import EnglishTextNormalizer
import os
import json
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize
import sys

import config

class CleaningModelBase:
    def __init__(self, text_normalizer):
        self.model = self._load_model()
        self.text_normalizer = text_normalizer

    def _load_model(self):
        pass

    def clean(self, references, hypotheses):
        pass

class CleaningModel(CleaningModelBase):
    TYPE_POS_NOUN = 'noun'
    TYPE_POS_VERB = 'verb'
    TYPE_POS_ADJ = 'adj'
    TYPE_POS_ADV = 'adv'
    TYPE_POS_NON_CONTENT = 'noncontent'
    TYPE_NAMED_ENTITY = 'named_entity'
    TYPE_POS_CONTENT = 'content'

    def __init__(self, text_normalizer, types_to_fix):
        super(CleaningModel, self).__init__(text_normalizer)
        self.is_type_funcs = []
        for type_to_fix in types_to_fix:
            if type_to_fix == CleaningModel.TYPE_POS_NOUN:
                self.is_type_funcs.append(self.__is_noun)
            elif type_to_fix == CleaningModel.TYPE_POS_VERB:
                self.is_type_funcs.append(self.__is_verb)
            elif type_to_fix == CleaningModel.TYPE_POS_ADJ:
                self.is_type_funcs.append(self.__is_adjective)
            elif type_to_fix == CleaningModel.TYPE_POS_ADV:
                self.is_type_funcs.append(self.__is_adverb)
            elif type_to_fix == CleaningModel.TYPE_POS_NON_CONTENT:
                self.is_type_funcs.append(self.__is_not_content_word)
            elif type_to_fix == CleaningModel.TYPE_NAMED_ENTITY:
                self.is_type_funcs.append(self.__is_named_entity)
            elif type_to_fix == CleaningModel.TYPE_POS_CONTENT:
                self.is_type_funcs.append(self.__is_content_word)

    def _load_model(self):
        self.nlp = spacy.load("en_core_web_sm")
        return None

    def __is_noun(self, pos_tag, is_ent):
        return pos_tag.startswith('NN')

    def __is_verb(self, pos_tag, is_ent):
        return pos_tag.startswith('VB')

    def __is_adjective(self, pos_tag, is_ent):
        return pos_tag.startswith('JJ')

    def __is_adverb(self, pos_tag, is_ent):
        return pos_tag.startswith('RB')

    def __is_not_content_word(self, pos_tag, is_ent):
        return not pos_tag.startswith('NN') and not pos_tag.startswith('VB') and not pos_tag.startswith('JJ') and not pos_tag.startswith('RB')
    
    def __is_named_entity(self, pos_tag, is_ent):
        return is_ent

    def __is_content_word(self, pos_tag, is_ent):
        return pos_tag.startswith('NN') or pos_tag.startswith('VB') or pos_tag.startswith('JJ') or pos_tag.startswith('RB')
        
    def __is_type_needed(self, pos_tag, is_ent, is_type_funcs):
        # if the given type atleast one of those needed, return True
        for is_type_func in is_type_funcs:
            if is_type_func(pos_tag, is_ent):
                return True
        return False

    def __rewrite_utterances_with_fix(self, references, hypotheses, is_type_funcs):
        
        # spacy objects for the texts:
        ref_docs = [self.nlp(ref) for ref in references]
        hyp_docs = [self.nlp(hyp) for hyp in hypotheses]

        # recreate the strings with the spacy word tokenization so that we can use the
        # same indices in the jiwer alignment:
        ref_strs = [' '.join([token.text for token in ref_doc]) for ref_doc in ref_docs]
        hyp_strs = [' '.join([token.text for token in hyp_doc]) for hyp_doc in hyp_docs]

        # create the sequence of POS tags for the two texts:
        refs_pos = [[token.tag_ for token in ref_doc] for ref_doc in ref_docs]
        hyps_pos = [[token.tag_ for token in hyp_doc] for hyp_doc in hyp_docs]

        # create the sequence of bolleans for whether a token is part of a named entity, for the two texts:
        refs_is_ent = [[token.ent_iob_ in ['B', 'I'] for token in ref_doc] for ref_doc in ref_docs]
        hyps_is_ent = [[token.ent_iob_ in ['B', 'I'] for token in hyp_doc] for hyp_doc in hyp_docs]
        
        hyp_strs_fixed = []
        for idx, (ref_str, hyp_str) in enumerate(zip(ref_strs, hyp_strs)):
            # if the ref is an empty string, just keep the original hyp string (also jiwer cannot process empty ref):
            if len(ref_str) == 0:
                hyp_strs_fixed.append(hyp_str)
                continue
            
            # get the edits between the two texts with respect to entities, content words and all words:
            alignment_obj = jiwer.process_words(ref_str, hyp_str)
            ref = alignment_obj.references[0]
            hyp = alignment_obj.hypotheses[0]
            align_chunks = alignment_obj.alignments[0]
            ref_pos = refs_pos[idx]
            hyp_pos = hyps_pos[idx]
            ref_is_ent = refs_is_ent[idx]
            hyp_is_ent = hyps_is_ent[idx]
            
            # add words one by one for the new hypothesis str:
            hyp_str_new_words = []
            for op in align_chunks:
                #print(op)
                if op.type == 'equal':
                    hyp_str_new_words.extend(hyp[op.hyp_start_idx: op.hyp_end_idx])  # keep as-is
                elif op.type == 'substitute':
                    for i, ref_idx in enumerate(range(op.ref_start_idx, op.ref_end_idx)): 
                        if self.__is_type_needed(ref_pos[ref_idx], ref_is_ent[ref_idx], is_type_funcs):
                            hyp_str_new_words.append(ref[ref_idx])  # fix this word since it has the relevant type in the ref
                        elif self.__is_type_needed(hyp_pos[op.hyp_start_idx + i], hyp_is_ent[op.hyp_start_idx + i], is_type_funcs):
                            hyp_str_new_words.append(ref[ref_idx])  # fix this word since it has the relevant type in the hyp
                        else:
                            hyp_str_new_words.append(hyp[op.hyp_start_idx + i])  # keep the original word in the hyp
                elif op.type == 'delete':
                    for ref_idx in range(op.ref_start_idx, op.ref_end_idx): 
                        if self.__is_type_needed(ref_pos[ref_idx], ref_is_ent[ref_idx], is_type_funcs):
                            hyp_str_new_words.append(ref[ref_idx])  # add this word since it has the relevant type
                elif op.type == 'insert':
                    for hyp_idx in range(op.hyp_start_idx, op.hyp_end_idx):
                        if not self.__is_type_needed(hyp_pos[hyp_idx], hyp_is_ent[hyp_idx], is_type_funcs):
                            hyp_str_new_words.append(hyp[hyp_idx])  # keep the original inserted word if it is not of the correct type (only remove if correct type)
            hyp_str_fixed = ' '.join(hyp_str_new_words)
            hyp_strs_fixed.append(hyp_str_fixed)
                
        return hyp_strs_fixed
    
    def clean(self, references, hypotheses):
        #hypotheses_cleaned, refs_norm, hyps_norm = self.__rewrite_utterances_with_fix(references, hypotheses, self.is_pos_funcs)
        hypotheses_cleaned = self.__rewrite_utterances_with_fix(references, hypotheses, self.is_type_funcs)
        return hypotheses_cleaned#, refs_norm, hyps_norm


class UtteranceNormalizerWhisper:
    def __init__(self) -> None:
        self.whisper_normalizer = EnglishTextNormalizer()

    def normalize(self, text):
        return self.whisper_normalizer(text)


class UtteranceNormalizerSplit:
    def __init__(self):
        pass

    def normalize(self, text):
        text = text.translate(str.maketrans('', '', ',.?!;:'))
        text = ' '.join(word_tokenize(text)).lower()
        text = ' ' + text  # add a space at the beginning to allow for easy replacement
        replacements = [
            ('okay', 'ok'),
            (" 'm", "'m"),
            (" 'll", "'ll"),
            (" 's", "'s"),
            (" 've", "'ve"),
            (" n't", "n't"),
            (' 0', 'oh')
        ]
        for r1, r2 in replacements:
            text = text.replace(r1, r2)
        return text.strip()


class TranscriptCleaner:
    def __init__(self, cleaning_model,
                 input_transcript_data_filepath, output_filepath, text_normalizer, 
                 chunk_size=20, debug=False):
        '''
        cleaning_model: an object based on CleaningModelBase which will be used to clean a transcript
        input_transcript_data_filepath: the transcript jsonl file (e.g., noised)
            set the path to the file. This is where the reference and predicted utterances are.
        output_filepath: where to output the generated transcript with the scores.
        '''
        self.debug = debug
        if self.debug:
            print('DEBUG MODE')
        self.data_transcript_filepath = input_transcript_data_filepath
        self.output_filepath = output_filepath
        self.model = cleaning_model
        self.conversation_id_to_utterances = self.__load_transcript_file(self.data_transcript_filepath)
        self.text_normalizer = text_normalizer
        self.chunk_size = chunk_size
        
    def __load_transcript_file(self, transcript_filepath):
        conversation_id_to_utterances = {}  # datum_id -> [{'id', 'ref', 'pred', 'idx', 'wer', 'cer'}]
        if transcript_filepath is not None:
            with open(transcript_filepath, 'r') as fIn:
                for line in fIn:
                    datum = json.loads(line.strip())
                    utt_id = datum['id']
                    conversation_id = '_'.join(utt_id.split('_')[:-1])
                    #utt_idx = int(utt_id.split('_')[-1])
                    #utt_text = datum['pred']
                    if conversation_id not in conversation_id_to_utterances:
                        conversation_id_to_utterances[conversation_id] = []
                    conversation_id_to_utterances[conversation_id].append(datum)
        return conversation_id_to_utterances

    def __compute_scores(self, utterances_pred, utterances_ref):
        scores_all = []
        for p, r in zip(utterances_pred, utterances_ref):
            p = self.text_normalizer.normalize(p)
            r = self.text_normalizer.normalize(r)
            if len(p) == 0 or len(r) == 0:
                wer = max(len(p.split()), len(r.split()))
                cer = max(len(p), len(r))
            else:
                wer = jiwer.wer(r, p)
                cer = jiwer.cer(r, p)
            scores_all.append({'wer': wer, 'cer': cer})
        return scores_all
    
    def __get_existing_data(self):
        if self.output_filepath and os.path.exists(self.output_filepath):
            conversation_id_to_utterances = self.__load_transcript_file(self.output_filepath)
        else:
            conversation_id_to_utterances = {}
        return conversation_id_to_utterances

    def __dump_existing_data(self, list_of_new_instances):
        # an utterance to write in the file example: {"id": "Bed003_67", "ref": "Nice coinage .", "pred": " Yes, boy.", "idx": 67, "wer": 1.0, "cer": 0.8333333333333334}
        if self.output_filepath:
            # dump the new instances created
            with open(self.output_filepath, 'a') as fOut:
                fOut.write('\n'.join([json.dumps(d) for d in list_of_new_instances]) + '\n')

    def __clean_transcript(self, transcript_utterances, transcript_utterances_reference):
        # loop chunk by chunk of the transcript:
        cleaned_utterances_all = []
        cleaned_utterances_all_scores = []
        utterances_all_scores_orig = []
        for i in tqdm(range(0, len(transcript_utterances), self.chunk_size)):
            # get a chunk of the transcript:
            chunk_utts = transcript_utterances[i: i + self.chunk_size]
            chunk_utts_ref = transcript_utterances_reference[i: i + self.chunk_size]
            chunk_utts_cleaned = self.model.clean(chunk_utts_ref, chunk_utts)
            cleaned_utterances_all.extend(chunk_utts_cleaned)

            # compute wer scores:
            chunk_scores = self.__compute_scores(chunk_utts_cleaned, chunk_utts_ref)
            chunk_scores_orig = self.__compute_scores(chunk_utts, chunk_utts_ref)
            cleaned_utterances_all_scores.extend(chunk_scores)
            utterances_all_scores_orig.extend(chunk_scores_orig)
        
        return cleaned_utterances_all, cleaned_utterances_all_scores, utterances_all_scores_orig
    
    def clean_transcript(self):
        conversation_id_to_utterances_processed = self.__get_existing_data()
        
        for datum_name in tqdm(self.conversation_id_to_utterances):
            if datum_name in conversation_id_to_utterances_processed:
                continue  # this datum has already been computed

            utts_noisy = [d['pred'] for d in self.conversation_id_to_utterances[datum_name]]
            utts_reference = [d['ref'] for d in self.conversation_id_to_utterances[datum_name]]
            utts_cleaned, utts_scores, utts_scores_orig = self.__clean_transcript(utts_noisy, utts_reference)
            new_data = []
            for utt_info_orig, utt_cleaned, utt_scores, utt_scores_orig in zip(self.conversation_id_to_utterances[datum_name], utts_cleaned, utts_scores, utts_scores_orig):
                utt_info_new = dict(utt_info_orig)
                utt_info_new['pred'] = utt_cleaned
                utt_info_new['wer'] = utt_scores['wer']
                utt_info_new['cer'] = utt_scores['cer']
                utt_info_new['wer_delta'] = utt_info_new['wer'] - utt_scores_orig['wer']
                utt_info_new['cer_delta'] = utt_info_new['cer'] - utt_scores_orig['cer']

                new_data.append(utt_info_new)
            
            self.__dump_existing_data(new_data)

            if self.debug:
                break

    def get_improvement_scores(self):
        wer_deltas = []
        cer_deltas = []
        with open(self.output_filepath) as fIn:
            for line in fIn:
                datum = json.loads(line.strip())
                wer_deltas.append(datum['wer_delta'])
                cer_deltas.append(datum['cer_delta'])

        results = {
            'wer': {
                'mean': np.mean(wer_deltas),
                'std': np.std(wer_deltas),
                'max': max(wer_deltas),
                'min': min(wer_deltas),
                'count': len(wer_deltas)
            },
            'cer': {
                'mean': np.mean(cer_deltas),
                'std': np.std(cer_deltas),
                'max': max(cer_deltas),
                'min': min(cer_deltas),
                'count': len(cer_deltas)
            }
        }
        return results

if __name__ == '__main__':
    # usage:
    # python3 run_cleaning_on_transcripts.py <dataset_id>
    # dataset_id: one of qmsum | qaconv | mrda
    #
    # Paths to set:
    #   BASE_PATH: the base path for this project
    BASE_PATH = config.BASE_PATH
    DEBUG = config.DEBUG

    assert len(sys.argv) == 2, 'Error: You must pass in one argument <qmsum|qaconv|mrda>'
    dataset_id = sys.argv[1]  # qmsum | qaconv | mrda
    assert dataset_id in ['qmsum', 'qaconv', 'mrda'], 'Error: Dataset can be one of qmsum, qaconv or mrda.'
    
    transcript_paths = [
        f'{BASE_PATH}/transcripts/{dataset_id}_test/whisper_small__{dataset_id}_clean_test.jsonl',
        f'{BASE_PATH}/transcripts/{dataset_id}_test___reverb__noise_10/whisper_small__{dataset_id}_noised_test_reverb_10.jsonl',
        f'{BASE_PATH}/transcripts/{dataset_id}_test___reverb__noise_5/whisper_small__{dataset_id}_noised_test_reverb_5.jsonl',
        f'{BASE_PATH}/transcripts/{dataset_id}_test___reverb__noise_0/whisper_small__{dataset_id}_noised_test_reverb_0.jsonl',
        f'{BASE_PATH}/transcripts/{dataset_id}_test___reverb__noise_-5/whisper_small__{dataset_id}_noised_test_reverb_m5.jsonl',
        f'{BASE_PATH}/transcripts/{dataset_id}_test___reverb__noise_-10/whisper_small__{dataset_id}_noised_test_reverb_m10.jsonl'
    ]
    text_normalizer = UtteranceNormalizerWhisper()

    for transcript_path in transcript_paths:
        for type_to_fix in [CleaningModel.TYPE_POS_NOUN, 
                            CleaningModel.TYPE_POS_VERB,
                            CleaningModel.TYPE_POS_ADJ,
                            CleaningModel.TYPE_POS_ADV,
                            CleaningModel.TYPE_POS_NON_CONTENT,
                            CleaningModel.TYPE_NAMED_ENTITY,
                            CleaningModel.TYPE_POS_CONTENT]:
            cleaning_model = CleaningModel(text_normalizer, [type_to_fix])
            output_filepath = f'{transcript_path[:-6]}_cleaned_{type_to_fix}.jsonl'
            cleaner = TranscriptCleaner(cleaning_model, transcript_path, output_filepath, 
                                        text_normalizer, debug=DEBUG)
            cleaner.clean_transcript()
            
            improvement_scores = cleaner.get_improvement_scores()
            print(output_filepath)
            print(improvement_scores)