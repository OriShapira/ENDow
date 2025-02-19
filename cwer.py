import spacy
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

class CWER:
    def __init__(self):
        # Load spaCy's English model
        self.nlp = spacy.load("en_core_web_sm")
        self.whisper_normalizer = EnglishTextNormalizer()
    
    def __normalize_utterance(self, text):
        text = self.whisper_normalizer(text)
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
        return text.strip()

    def __get_alignment_info(self, references, hypotheses):
        # normalize the utterances and get rid of instances with empty references (it causes an error in jiwer.process_words):
        references_norm = []
        hypotheses_norm = []
        for r, h in zip(references, hypotheses):
            r_n = self.__normalize_utterance(r)
            h_n = self.__normalize_utterance(h)
            if len(r_n) > 0:
                references_norm.append(r_n)
                hypotheses_norm.append(h_n)
        
        # spacy objects for the texts:
        ref_docs = [self.nlp(ref) for ref in references_norm]
        hyp_docs = [self.nlp(hyp) for hyp in hypotheses_norm]

        # recreate the strings with the spacy word tokenization so that we can use the
        # same indices in the jiwer alignment:
        ref_strs = [' '.join([token.text for token in ref_doc]) for ref_doc in ref_docs]
        hyp_strs = [' '.join([token.text for token in hyp_doc]) for hyp_doc in hyp_docs]

        # get alignment information between the two texts (insertions, deltions, substitutions)
        alignment_obj = jiwer.process_words(ref_strs, hyp_strs)

        # get the edits between the two texts:
        alignments_info = []
        for utt_idx, (ref, pred, align_chunks) in enumerate(
            zip(alignment_obj.references, 
                alignment_obj.hypotheses,
                alignment_obj.alignments)):
        
            # add more information to the alignment chunks from jiwer:
            for op in align_chunks:
                ref_s = op.ref_start_idx
                ref_e = op.ref_end_idx
                pred_s = op.hyp_start_idx
                pred_e = op.hyp_end_idx

                ref_words = ref[ref_s: ref_e]
                pred_words = pred[pred_s: pred_e]

                alignments_info.append({'ref_span': ' '.join(ref_words),
                                        'hyp_span': ' '.join(pred_words),
                                        'op_type': op.type,
                                        'utt_idx': utt_idx})
                
        return ref_strs, hyp_strs, alignment_obj, alignments_info

    def __compute_instance(self, references, hypotheses, get_alignment=False):
        try:
            ref_str, hyp_str, alignment_obj, alignments_info = \
                self.__get_alignment_info(references, hypotheses)

            scores = {'mer':  alignment_obj.mer,
                      'wil':  alignment_obj.wil,
                      'wip':  alignment_obj.wip,
                      'hits':  alignment_obj.hits,
                      'substitutions':  alignment_obj.substitutions,
                      'insertions': alignment_obj.insertions,
                      'deletions': alignment_obj.deletions,
                      'wer': alignment_obj.wer,
                      'cer': jiwer.cer(ref_str, hyp_str)}

        except Exception as ex:
            raise ex
        
        if get_alignment:
            return {'scores': scores, 'alignments': alignments_info}
        else:
            return scores
    
    
    def compute(self, predictions, references, compute_overall=False, get_alignment=False):
        assert isinstance(predictions, list) and isinstance(references, list), 'Variables `predictions` and `references` must be lists.'
        assert len(predictions) == len(references), 'The lists of predictions and references do not have the same length.'

        if not compute_overall:
            outputs = []
            for p, r in zip(predictions, references):
                output = self.__compute_instance([r], [p], get_alignment=get_alignment)
                outputs.append(output)
            return outputs
        else:
            output = self.__compute_instance(references, predictions, get_alignment=get_alignment)
            return output
        


if __name__ == '__main__':
    cwer = CWER()

    #reference =  "Barack Obama was the forty fourth President of the United States."
    #hypothesis = "Barack Obama was the for the hour president of these united slate last year."

    #res = cwer.compute([hypothesis], [reference], get_alignment=True)

    reference =  ["Barack Obama was the forty fourth President of the United States.", "hello world."]
    hypothesis = ["Barack Obama was the for the hour president of these united slate last year.", "hello man"]

    res = cwer.compute(hypothesis, reference, compute_overall=True, get_alignment=True)
    print(res)