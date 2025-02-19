"""
Evaluation script for QAConv based on https://github.com/salesforce/QAConv/blob/master/evaluate.py
but changed for required usage.
"""

import collections
import re
import string
from fuzzywuzzy import fuzz
from num2words import num2words
from word2number import w2n
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import scipy.stats as stats


def __normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def __get_tokens(s_normalized):
    if not s_normalized: return []
    return s_normalized.split()


def __compute_exact(a_gold_normalized, a_pred_normalized):
    return int(a_gold_normalized == a_pred_normalized)


def __compute_fuzz_ratio(a_gold_normalized, a_pred_normalized):
    return fuzz.ratio(a_pred_normalized, a_gold_normalized)


def __add_word_number_mapping(answers_normalized):
    added_ans = []
    for ans in answers_normalized:
        if ans.isdigit():
            added_ans.append(__normalize_answer(num2words(ans)))
        else:
            try:
                temp = __normalize_answer(str(w2n.word_to_num(ans)))
                added_ans.append(temp)
            except:
                pass
    return added_ans


def __compute_unanswerable_binary(a_gold_list_normalized, a_pred_list_normalized):
    gold_unanswerable_bin = [1 if "unanswerable" in a else 0 for a in a_gold_list_normalized]
    pred_unanswerable_bin = [1 if a == "unanswerable" else 0 for a in a_pred_list_normalized]
    f1_stats = __compute_unanswerable_binary_from_bin_lists(gold_unanswerable_bin, pred_unanswerable_bin)
    return f1_stats, gold_unanswerable_bin, pred_unanswerable_bin


def __compute_unanswerable_binary_from_bin_lists(gold_list_bin, pred_list_bin):
    # give lists of 0s and 1s and get the binary F1
    prec, rec, f1, _ = precision_recall_fscore_support(gold_list_bin, pred_list_bin, zero_division=1.0, average='binary')
    return {'precision': prec, 'recall': rec, 'f1': f1}

def __compute_f1(a_gold_normalized, a_pred_normalized):
    gold_toks = __get_tokens(a_gold_normalized)
    pred_toks = __get_tokens(a_pred_normalized)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = float(num_same) / len(pred_toks)
    recall = float(num_same) / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def __get_raw_scores(answers_ref, answers_pred):
    '''
    answers_ref: list of lists of answers (each qa instance might have several reference answers)
    answers_pred: list of answers as predicted by a model, in the same question-order as answers_ref

    returns exact_scores, f1_scores, fuzz_ratio_scores, unanswerable_binary_f1, a_gold_unanswerable, a_pred_unanswerable
        exact_scores: list of exact scores, in the order of the input answers
        f1_scores: list of token-level F1 scores, in the order of the input answers
        fuzz_ratio_scores: list of fuzzy-match scores, in the order of the input answers
        unanswerable_binary_f1: overall F1 binary score for unanswerable correctness
        a_gold_unanswerable: list of 0s and 1s whether the answer is unanswerable or not in the gold answers
        a_pred_unanswerable: list of 0s and 1s whether the answer is unanswerable or not in the predicted answers
    '''
    exact_scores = []
    f1_scores = []
    fuzz_ratio_scores = []
    a_gold_list_normalized, a_pred_list_normalized = [], []
    for a_ref, a_pred in zip(answers_ref, answers_pred):
        gold_answers_normalized = [__normalize_answer(a) for a in a_ref]
        gold_answers_normalized += __add_word_number_mapping(gold_answers_normalized)
        if not gold_answers_normalized:
            # For unanswerable questions, only correct answer is empty string
            gold_answers_normalized = ['unanswerable']
        
        pred_answer_normalized = __normalize_answer(a_pred)
        
        # Take max over all gold answers
        exact_scores.append(max(__compute_exact(a, pred_answer_normalized) for a in gold_answers_normalized))
        f1_scores.append(max(__compute_f1(a, pred_answer_normalized) for a in gold_answers_normalized))
        fuzz_ratio_scores.append(max(__compute_fuzz_ratio(a, pred_answer_normalized) for a in gold_answers_normalized))
        
        a_gold_list_normalized.append(gold_answers_normalized)
        a_pred_list_normalized.append(pred_answer_normalized)
    
    unanswerable_binary_f1_stats, a_gold_unanswerable, a_pred_unanswerable = __compute_unanswerable_binary(a_gold_list_normalized, a_pred_list_normalized)

    return exact_scores, f1_scores, fuzz_ratio_scores, unanswerable_binary_f1_stats, a_gold_unanswerable, a_pred_unanswerable


def __get_stats(values_list):
    # Compute the confidence interval according to the confidence_level specified.
    # E.g., for confidence_level=0.95, the confidence interval is for the (0.025, 0.975) percentiles.
    # The assumption is that the list of values converges to a normal distibution.
    confidence_level=0.95
    mean = np.mean(values_list)
    sem = stats.sem(values_list)  # Standard Error of the Mean
    n = len(values_list)
    critical_value = stats.t.ppf((1 + confidence_level) / 2., n - 1)  # t-distribution
    margin_of_error = critical_value * sem
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return {
        'mean': mean, 
        'std': np.std(values_list), 
        'n': n, 
        'confidence_interval_95': confidence_interval, 
        'min': min(values_list), 
        'max': max(values_list),
        'median': np.median(values_list),
        'quartile_1': np.percentile(values_list, 25),
        'quartile_3': np.percentile(values_list, 75),
    }


def make_eval_dicts(exact_scores, f1_scores, fuzz_scores, unans_bin_f1_stats, unans_list_gold, unans_list_pred):
    # if unans_bin_f1_score is None, it is computed based on the unans_list_gold and unans_list_pred binary lists
    overall_scores = {
        'exact': __get_stats(exact_scores),
        'f1':  __get_stats(f1_scores),
        'fzr':  __get_stats(fuzz_scores),
        'unans_f1': unans_bin_f1_stats if unans_bin_f1_stats is not None \
                                       else __compute_unanswerable_binary_from_bin_lists(unans_list_gold, unans_list_pred)
    }
    all_scores = {
        'exact': exact_scores,
        'f1': f1_scores,
        'fzr': fuzz_scores,
        'unans_gold': unans_list_gold,
        'unans_pred': unans_list_pred
    }
    return overall_scores, all_scores
    

def evaluate_answers(answers_ref, answers_pred):
    '''
    answers_ref: list of lists of answers (each qa instance might have several reference answers)
    answers_pred: list of answers as predicted by a model, in the same question-order as answers_ref

    returns overall_scores, all_scores
        overall_scores: a dictionary with metrics exact, f1, fzr, unans_f1, where each of the first three metrics have a dictionary of stats.
        all_scores: a dictionry with metrics exact, f1, fzr, each with the lists of scores for each of predicted answers (in the same order), and also
            the unans_gold and unans_pred lists.
    '''
    exact_raw, f1_raw, fuzz_raw, unans_bin_f1_stats, unans_list_gold, unans_list_pred = __get_raw_scores(answers_ref, answers_pred)
    overall_scores, all_scores = make_eval_dicts(exact_raw, f1_raw, fuzz_raw, unans_bin_f1_stats, unans_list_gold, unans_list_pred)
    return overall_scores, all_scores


if __name__ == '__main__':
    answers_ref = [
        ['120 hours'],
        ['The best time is now.'],
        ['October 25, 2025', 'next Wednseday'],
        ['the main character is harry', 'Harry Potter', 'the child wizard'],
        [],
        ['20']
    ]
    answers_pred = [
        '120',
        'the time is now',
        'Wednesday the 25th',
        'harry potter',
        'unanswerable',
        'unanswerable'
    ]
    overall_scores, all_scores = evaluate_answers(answers_ref, answers_pred)
    print(overall_scores)
    print(all_scores)