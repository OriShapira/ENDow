import os
import sys
import json
from cwer import CWER
import numpy as np
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import config



class ResultsDisplayer:
    # A class for displaying the results of WER vs. task results.
    # Creates a plot comparing baseline (no cleaning) lines for different models, and a plot comparing a
    # baseline to its cleaned versions.
    #
    # Usage example:
    #   > results_displayer = ResultsDisplayer(base_dir, data_folder_names_ordered, task_name, task_model_names, 
    #                                          task_metrics, cleaning_versions_to_show, audio_metrics)
    #   > results_displayer.execute()

    def __init__(self, base_dir, data_folder_names_ordered, task_name, task_model_names, 
                 task_metrics, output_folder, cleaning_versions_to_show=None, 
                 audio_metrics=['wer'], output_figures=True):
        
        # base_dir: the base_dir from which all the data and results can be read
        # data_folder_names_ordered: A list of pairs of (<folder_name_for_noise>, <noise_name_for_folder>)
        # task_name: for displaying purposes only (mrda, qaconv, qmsum)
        # task_model_names: the list of model names for which to display plots (e.g., Mistral7BInstruct...)
        # task_metrics: the list of task metrics for which to show results
        # cleaning_versions_to_show: the cleaning versions to show on a plot (each cleaning version is a line). If None, shows
        #    all, otherwise a list can be provided with any of ('', 'cleaned_noun', 'cleaned_verb', 'cleaned_adj', 'cleaned_adv', 'cleaned_noncontent', 'cleaned_content', 'cleaned_named_entities')
        #    The '' version is the original results, without any extra cleaning conducted on the noised transcripts.
        # audio_metrics: list of audio metrics for which to display results

        self.cwer = CWER()  # for computing noise scores on transcripts
        
        self.base_dir = base_dir
        self.data_folder_names_ordered = data_folder_names_ordered
        self.task_name = task_name
        self.task_model_names = task_model_names
        self.task_metrics = task_metrics
        self.cleaning_versions_to_show = cleaning_versions_to_show
        self.audio_metrics = audio_metrics
        self.output_folder = output_folder
        self.output_figures = output_figures

    
    def _prepare_scores_for_display(self):
        # model_version_to_downstream_results:  noise_name -> model_version_id -> clean_version_id -> metric -> {<stat>: <val>}
        #   noise_name is the level of noise added (original, none, 10, 5, 0, -5, -10)
        #   model_version_id is the summarization model (e.g. Llama3Instruct_recursive)
        #   clean_version_id is the type of cleaning done to the transcript (e.g. cleaned_noun, or '' if none)
        # noise_name_to_clean_ver_to_cwer_scores:  noise_name -> clean_version_id -> metric -> (<mean>, <std>)
        #   noise_name is the level of noise added (original, none, 10, 5, 0, -5, -10)
        #   clean_version_id is the type of cleaning done to the transcript ('', 'cleaned_noun', 'cleaned_verb', 'cleaned_adj', 'cleaned_adv', 'cleaned_noncontent', 'cleaned_content', 'cleaned_named_entities')
        
        self.noise_name_to_clean_ver_to_cwer_scores = {}
        self.noise_name_to_version_to_downstream_results = {}
        self.noise_names_ordered = []
        for data_folder_name, noise_name in self.data_folder_names_ordered:
            print(data_folder_name)
            clean_ver_to_cwer_scores, model_version_to_downstream_results = self._get_all_scores(data_folder_name)
            self.noise_name_to_clean_ver_to_cwer_scores[noise_name] = clean_ver_to_cwer_scores
            self.noise_name_to_version_to_downstream_results[noise_name] = model_version_to_downstream_results
            self.noise_names_ordered.append(noise_name)

    
    def execute(self):
        self._prepare_scores_for_display()

        print('--- BASELINES ---')
        task_model_to_task_metric_to_curve_scores_baselines = {}  # task_model_name -> task_metric -> {'auc', 'auc_interval', 'ntp'}
        for task_metric in self.task_metrics:
            print(f'--------------\n--------------\n{task_metric}\n--------------\n')
            task_model_to_curve_scores = self._plot_results_baselines(task_metric=task_metric, audio_metric='wer')
            for task_model in task_model_to_curve_scores:
                if task_model not in task_model_to_task_metric_to_curve_scores_baselines:
                    task_model_to_task_metric_to_curve_scores_baselines[task_model] = {}
                task_model_to_task_metric_to_curve_scores_baselines[task_model][task_metric] = task_model_to_curve_scores[task_model]

        print('\n\n\n--- CLEANING ---')
        task_model_to_task_metric_to_cleaning_version_to_curve_scores = {}  # task_model_name -> task_metric -> cleaning_version -> {'ces'}
        for task_model_name in self.task_model_names:
            task_model_to_task_metric_to_cleaning_version_to_curve_scores[task_model_name] = {}
            for audio_metric in self.audio_metrics:
                for task_metric in self.task_metrics:
                    print(f'--------------\n--------------\n{task_model_name} - {audio_metric} - {task_metric}\n--------------\n')
                    cleaning_version_to_curve_scores = self._plot_results_cleaning(task_model_name=task_model_name, 
                                                                                   task_metric=task_metric, 
                                                                                   audio_metric=audio_metric)
                    if audio_metric == 'wer':
                        task_model_to_task_metric_to_cleaning_version_to_curve_scores[task_model_name][task_metric] = {}
                        for cleaning_version in cleaning_version_to_curve_scores:
                            task_model_to_task_metric_to_cleaning_version_to_curve_scores[task_model_name][task_metric][cleaning_version] = \
                                cleaning_version_to_curve_scores[cleaning_version]
                            
        return task_model_to_task_metric_to_curve_scores_baselines, task_model_to_task_metric_to_cleaning_version_to_curve_scores

    
    def _get_cwer_scores(self, input_transcription_filepath, output_cwer_filepath):
        # computes the CWER scores for thethe transcription file specified, and puts the scores in the output file
        # path specified. If the output file already exists and has all the scores, then it is not recomputed.

        # if already computed, get it from the file:
        if os.path.exists(output_cwer_filepath):
            with open(output_cwer_filepath) as fIn:
                cwer_overall = json.load(fIn)['overall']
        # otherwise compute from scratch:
        else:
            conversation_id_to_utts = {}  # conv_id -> list of {'ref': <str>, 'pred': <str>}
            with open(input_transcription_filepath, 'r') as fIn:
                for line in fIn:
                    utterance_info = json.loads(line.strip())
                    utt_id = utterance_info['id']
                    conversation_id = '_'.join(utt_id.split('_')[:-1])
                    #utt_idx = int(utt_id.split('_')[-1])
                    if conversation_id not in conversation_id_to_utts:
                        conversation_id_to_utts[conversation_id] = []
                    conversation_id_to_utts[conversation_id].append({'ref': utterance_info['ref'], 'pred': utterance_info['pred']})


            # compute overall error rates on each conversation:
            conversation_id_to_cwer = {}  # cwer scores dicts (for each of the conversations)
            for conversation_id in tqdm(conversation_id_to_utts):
                all_refs = [u['ref'] for u in conversation_id_to_utts[conversation_id]]
                all_preds = [u['pred'] for u in conversation_id_to_utts[conversation_id]]
                cwer_result = self.cwer.compute(all_preds, all_refs, compute_overall=True)
                #print(cwer_result)
                conversation_id_to_cwer[conversation_id] = cwer_result

            # aggregate the scores for each CWER metric over all conversations:
            metric_scores = {}  # metric -> list of values from all conversations
            for conversation_id, cwer_result in conversation_id_to_cwer.items():
                for metric in cwer_result:
                    if metric not in metric_scores:
                        metric_scores[metric] = []
                    metric_scores[metric].append(cwer_result[metric])
            
            # metric -> (<mean>, <std>)   overall cwer scores over all conversations:
            cwer_overall = {metric: (np.mean(scores), np.std(scores)) for metric, scores in metric_scores.items()}

            # save the CWER scores to the file:
            info_to_save = {'conversations': conversation_id_to_cwer, 'overall': cwer_overall}
            with open(output_cwer_filepath, 'w') as fOut:
                json.dump(info_to_save, fOut, indent=4)

        return cwer_overall


    def _get_downstream_scores(self, input_results_filepath):
        # gets the task scores from the results file specified
        downstream_results = {}
        try:
            with open(input_results_filepath, 'r') as fIn:
                results_info = json.load(fIn)
                # in the QMSum results there is an 'all' field under the 'overall' field:
                if 'all' in results_info['overall']:
                    for metric in results_info['overall']['all']:
                        downstream_results[metric] = results_info['overall']['all'][metric]
                else:
                    for metric in results_info['overall']:
                        downstream_results[metric] = results_info['overall'][metric]
            return downstream_results
        except:
            return {}


    def _get_all_scores(self, data_folder_name):
        # gets the WER and task scores for the results folder specified

        model_name_to_downstream_results = {}  # model_name -> clean_version_id -> metric -> {<stat>: <val>}   where model_name is the task model (e.g. Llama3Instruct) and clean_version_id is the type of cleaning done to the transcript (e.g. cleaned_noun, or '' if none)
        clean_ver_to_cwer_scores = {}  # clean_version_id -> metric -> (<mean>, <std>)   where clean_version_id is the type of cleaning done to the transcript (e.g. cleaned_noun)
        output_cwer_filepath_template = os.path.join(self.base_dir, 'results', data_folder_name, 'cwer_resultsCLEANVERSIONID.json')

        transcripts_folder_path = os.path.join(self.base_dir, 'transcripts', data_folder_name)
        if os.path.exists(transcripts_folder_path):  # doesn't exist for the source (reference) transcripts
            for filename in tqdm(os.listdir(transcripts_folder_path)):
                # a transcription jsonl file:
                if filename.startswith('whisper_small') and filename.endswith('.jsonl'):
                    input_transcription_filepath = os.path.join(transcripts_folder_path, filename)
                    if '_cleaned_' in filename:
                        clean_ver_id_char_idx = filename.index('_cleaned_')
                        clean_ver_id = filename[clean_ver_id_char_idx: -6]  # e.g. "whisper_small__qmsum_clean_test_cleaned_adj.jsonl" -> "_cleaned_adj"
                    else:
                        clean_ver_id = ''
                    output_cwer_filepath = output_cwer_filepath_template.replace('CLEANVERSIONID', clean_ver_id)
                    cwer_overall = self._get_cwer_scores(input_transcription_filepath, output_cwer_filepath)
                    clean_ver_to_cwer_scores[clean_ver_id[1:]] = cwer_overall  # (remove trailing '_' from e.g. _cleaned_adj)
                
        results_folder_path = os.path.join(self.base_dir, 'results', data_folder_name)
        for filename in tqdm(os.listdir(results_folder_path)):
            # a task results file:
            if filename.startswith('results') and filename.endswith('json'):
                model_name = filename.replace('results_', '')[:-5]  # e.g. "results_Llama3Instruct.json" -> "Llama3Instruct"
                if '_cleaned_' in model_name:
                    clean_ver_id_char_idx = model_name.index('_cleaned_')
                    clean_ver_id = model_name[clean_ver_id_char_idx + 1:]  # e.g. "Llama3Instruct_cleaned_adj" -> "cleaned_adj"
                    model_name = model_name[:clean_ver_id_char_idx]  # e.g. "Llama3Instruct_cleaned_adj" -> "Llama3Instruct"
                else:
                    clean_ver_id = ''

                input_results_filepath = os.path.join(results_folder_path, filename)
                if model_name not in model_name_to_downstream_results:
                    model_name_to_downstream_results[model_name] = {}
                model_name_to_downstream_results[model_name][clean_ver_id] = self._get_downstream_scores(input_results_filepath)

        return clean_ver_to_cwer_scores, model_name_to_downstream_results


    def _plot_results_baselines(self, task_metric, audio_metric='wer'):
        # Draws the graph with the baseline lines (when no cleaning is conducted) for all the task models in self.task_model_names.
        # Puts the AUC and noise-tolerance point for each curve.
        # The graph is saved to "<self.output_folder>/{self.task_name}_noclean_{task_metric}.pdf"
        
        task_model_to_curve_scores = {}

        markers = 'os^D'
        for i, task_model_name in enumerate(self.task_model_names):
            x_list, y_list, y_err_lower_list, y_err_upper_list = self._get_xy_from_scores_no_clean(task_model_name, task_metric, audio_metric)
            noise_toleration_point = self._find_first_significant_change_in_y(list(reversed(x_list)), list(reversed(y_list)), 
                                                                              list(reversed(y_err_upper_list)), list(reversed(y_err_lower_list)))
            noise_toleration_point_str = f'{noise_toleration_point[0]:.3f}' if noise_toleration_point else "n/a"
            auc = np.trapz(list(reversed(y_list)), list(reversed(x_list)))
            auc_lower = np.trapz(list(reversed(y_err_lower_list)), list(reversed(x_list)))
            auc_upper = np.trapz(list(reversed(y_err_upper_list)), list(reversed(x_list)))
            auc_interval = max((auc - auc_lower), (auc_upper - auc))
            label_name = task_model_name.replace('_recursive', '').replace('_truncate', '').replace('Instruct', '').replace('_', '.').replace('7B', '')
            label_name += f'; AUC: {auc:.2f} ± {auc_interval:.2f}; NTP: {noise_toleration_point_str}'
            plt.plot(x_list, y_list, label=label_name, marker=markers[i], markersize=5, linestyle='-')  # color='blue'
            plt.fill_between(x_list, y_err_lower_list, y_err_upper_list, alpha=0.3) #color='blue', , label='Confidence Interval')
            if noise_toleration_point:
                plt.plot(noise_toleration_point[0], noise_toleration_point[1], '.', color='black')
                #plt.plot([critical_noise_point[0], critical_noise_point[0]], [0, critical_noise_point[1]], linestyle='--', color='black', linewidth=0.8)
            task_model_to_curve_scores[task_model_name] = {'auc': auc, 'auc_interval': auc_interval, 'ntp': noise_toleration_point}
        
        if self.output_figures:
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["figure.figsize"] = (10,5)
            plt.title(f'Model Performance @ Noise Level')
            plt.xlabel(audio_metric)
            plt.ylabel(task_metric)
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.output_folder, f"{self.task_name}_noclean_{task_metric}.pdf"),
                        format="pdf", bbox_inches="tight")
            plt.show()
        plt.clf()

        return task_model_to_curve_scores


    def _find_first_significant_change_in_y(self, x, y, y_uci, y_lci):
        # Finds the first point where the upper confidence value is equal to the lower confidence
        # value of the first point on the line (i.e. first significant decrease in y value).
        target_y = y_lci[0]  # The first y-lower-confidence-interval value
        for i in range(len(x) - 1):
            # Check if the segment contains the target value
            if y_uci[i] <= target_y <= y_uci[i + 1] or y_uci[i] >= target_y >= y_uci[i + 1]:
                # Linearly interpolate between points
                if y_uci[i] != y_uci[i + 1]:  # Prevent division by zero
                    interp_x = x[i] + (target_y - y_uci[i]) * (x[i + 1] - x[i]) / (y_uci[i + 1] - y_uci[i])
                    interp_y = y[i] + (interp_x - x[i]) / (x[i+1] - x[i]) * (y[i+1] - y[i])
                    if x[i] <= interp_x <= x[i + 1]:  # Ensure the point is on the correct segment
                        return (interp_x, interp_y)
        return None  # Return None if no point is found
    

    def _get_xy_from_scores_no_clean(self, task_model_name, task_metric, audio_metric):
        # Gets the WER and task scores as a lists of X and Y values as well as with the Y confidence intervals,
        # for the line with no cleaning (cleaning_version == ''). Uses the task_metric and audio_metric specified
        # for the XY values, and for the task_model_name specified.
        # Returns x_list, y_list, y_err_lower_list, y_err_upper_list
        return self._get_xy_from_scores(task_model_name, '', task_metric, audio_metric)


    def _get_xy_from_scores(self, task_model_name, cleaning_version, task_metric, audio_metric):
        # Gets the WER and task scores as a lists of X and Y values as well as with the Y confidence intervals,
        # for the line with the cleaning version specified. Uses the task_metric and audio_metric specified
        # for the XY values, and for the task_model_name specified.
        # Returns x_list, y_list, y_err_lower_list, y_err_upper_list
        x_list = []
        y_list = []
        y_err_lower_list = []
        y_err_upper_list = []
        for noise_name in self.noise_names_ordered:
            # the y-values are the task score:
            # the QMSum, QAConv, MRDA standard metrics are in the format: noise_name -> task_model_name -> cleaning_method -> metric -> <scores info>
            #if noise_name in self.noise_name_to_version_to_downstream_results:
            
            # in MRDA, there is no mean for the metric, but just a score
            # for QMSum and QAConv there is a mean for the metric
            score_key_name = 'score' if 'score' in self.noise_name_to_version_to_downstream_results[noise_name][task_model_name][''][task_metric] else 'mean'

            if noise_name != 'original':
                x_list.append(self.noise_name_to_clean_ver_to_cwer_scores[noise_name][cleaning_version][audio_metric][0])
                y_list.append(self.noise_name_to_version_to_downstream_results[noise_name][task_model_name][cleaning_version][task_metric][score_key_name])
                y_err_lower_list.append(self.noise_name_to_version_to_downstream_results[noise_name][task_model_name][cleaning_version][task_metric]['confidence_interval_95'][0])
                y_err_upper_list.append(self.noise_name_to_version_to_downstream_results[noise_name][task_model_name][cleaning_version][task_metric]['confidence_interval_95'][1])
            else:  # for the original transcript (no noise at all), there is no cleaning_version, and the WER is always zero, and we need to use the task scores of the '' cleaning version
                x_list.append(0)
                y_list.append(self.noise_name_to_version_to_downstream_results[noise_name][task_model_name][''][task_metric][score_key_name])
                y_err_lower_list.append(self.noise_name_to_version_to_downstream_results[noise_name][task_model_name][''][task_metric]['confidence_interval_95'][0])
                y_err_upper_list.append(self.noise_name_to_version_to_downstream_results[noise_name][task_model_name][''][task_metric]['confidence_interval_95'][1])

        return  x_list, y_list, y_err_lower_list, y_err_upper_list


    def _line_design_for_cleaning_technique(self, cleaning_technique):
        marker_map = {
            '': 'o',
            'cleaned_noun': '^',
            'cleaned_verb': '>',
            'cleaned_adj': 'v',
            'cleaned_adv': '<',
            'cleaned_noncontent': 's',
            'cleaned_content': 'D',
            'cleaned_named_entity': 'd'
        }
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_map = {
            '': color_cycle[0],
            'cleaned_noun': color_cycle[1],
            'cleaned_verb': color_cycle[2],
            'cleaned_adj': color_cycle[3],
            'cleaned_adv': color_cycle[4],
            'cleaned_noncontent': color_cycle[5],
            'cleaned_content': color_cycle[6],
            'cleaned_named_entity': color_cycle[7]
        }
        return marker_map[cleaning_technique], color_map[cleaning_technique]

    def _plot_results_cleaning(self, task_model_name, task_metric, audio_metric):
        # Draws the graph with the baseline line (when no cleaning is conducted) and the lines for the cleaned versions
        # of the baseline. Does so for the task_model_name, task_metric, audio_metric specified.
        # Puts the cleaning-effectiveness score for each cleaning version.
        # The graph is saved to "<self.output_folder>/{self.task_name}_cleaning_{task_model_name}_{task_metric}.pdf"

        # which cleaning versions to show (if None, use all the available ones):
        if self.cleaning_versions_to_show is None:
            self.cleaning_versions_to_show = list(self.noise_name_to_clean_ver_to_cwer_scores.values())[0]

        # place the line of the baseline (no cleaning):
        x_list_baseline, y_list_baseline, y_err_lower_list_baseline, y_err_upper_list_baseline = \
            self._get_xy_from_scores_no_clean(task_model_name, task_metric, audio_metric)
        marker, color = self._line_design_for_cleaning_technique('')
        plt.plot(x_list_baseline, y_list_baseline, label='no_cleaning', marker=marker, markersize=5, linestyle='-', color=color)
        plt.fill_between(x_list_baseline, y_err_lower_list_baseline, y_err_upper_list_baseline, alpha=0.3, color=color)
        
        cleaning_version_to_curve_scores = {}
        # for each of the cleaning methods (excluding the baseline), plot the curve and get the effectiveness score:
        for cleaning_version in self.cleaning_versions_to_show:
            if cleaning_version != '':  # we already processed and drew the baseline (not cleaned) line above
                x_list, y_list, y_err_lower_list, y_err_upper_list = \
                    self._get_xy_from_scores(task_model_name, cleaning_version, task_metric, audio_metric)
                
                ces = self._get_cleaning_effectiveness_score(x_list, y_list, x_list_baseline, y_list_baseline)
                label_name = cleaning_version.replace('cleaned_', '')
                label_name += f'; CES: {ces:.2f}'
                marker, color = self._line_design_for_cleaning_technique(cleaning_version)
                plt.plot(x_list, y_list, label=label_name, marker=marker, markersize=5, linestyle='-', color=color)
                plt.fill_between(x_list, y_err_lower_list, y_err_upper_list, alpha=0.3, color=color)
                #for i, txt in enumerate(noise_names_ordered):
                #    plt.annotate(txt, (x_list[i], y_list[i]), textcoords="offset points", xytext=(5, 5), ha='center')
                cleaning_version_to_curve_scores[cleaning_version] = {'ces': ces}
            
        if self.output_figures:
            plt.rcParams.update({'font.size': 12})
            plt.rcParams["figure.figsize"] = (10,5)
            label_task_model_name = task_model_name.replace('_recursive', '').replace('_truncate', '').replace('Instruct', '').replace('_', '.')
            plt.title(f'{label_task_model_name} Perfomance with Cleaning @ Noise Level')
            plt.xlabel(audio_metric)
            plt.ylabel(task_metric)
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.output_folder, f"{self.task_name}_cleaning_{task_model_name}_{task_metric}.pdf"), 
                        format="pdf", bbox_inches="tight")
            plt.show()
        plt.clf()

        return cleaning_version_to_curve_scores


    def _get_cleaning_effectiveness_score(self, x_l_cleaned, y_l_cleaned, x_l_baseline, y_l_baseline):
        # Computes a "cleaning effectiveness score" which is a function of the pointwise changes between the baseline
        # line and the line for a cleaning version.

        # get the y value at x=0, to normalize with it the change in a y
        y_0_baseline = y_l_baseline[x_l_baseline.index(0)]
        # get the list of pointwise changes (f1(delta_y) / f2(delta_x))
        change_list = []
        for x_b, y_b, x_c, y_c in zip(x_l_baseline, y_l_baseline, x_l_cleaned, y_l_cleaned):
            # make sure the change in x is not zero (add epsilon), and normalize as sqrt
            delta_x = (x_b - x_c + 0.000001) ** (0.5)
            # normalize the change in y with the y_value at x=0, so that it is the relative change with respect to the initial value
            delta_y = (y_c - y_b) / y_0_baseline
            '''
            # OTHER OPTIONS
            # delta_x = (x_b - x_c + 0.000001)  # no normalization for x
            # delta_y = 10 * ((y_c - y_b) / y_0_baseline)  # linear increase of delta_y
            # delta_x = -math.log(x_b - x_c + 0.000001)  # negative log normalization for delta_x
            '''
            #print(delta_x, delta_y)
            change_list.append(delta_y / delta_x)
        # the final score is the average of the pointwise changes:
        avg_change = np.mean(change_list)
        return avg_change
    


class ResultsDisplayerPairwiseRanking(ResultsDisplayer):
    # A class for displaying the results of WER vs. task results, specifically for Pairwise Ranking socres on QMSum.
    # For the standard metrics, the base class ResultsDisplayer is needed.
    # Creates a plot comparing baseline (no cleaning) lines for different models, and a plot comparing a
    # baseline to its cleaned versions.

    def __init__(self, base_dir, data_folder_names_ordered, task_name, task_model_names, task_metrics, output_folder,
                 pairwise_eval_folderpath_change, pairwise_eval_folderpath_base, 
                 cleaning_versions_to_show=None, audio_metrics=['wer'], output_figures=True):
        
        # See the explanations of the parameters in the constuctor of the base class.
        # Extra parameters here:
        # pairwise_eval_folderpath_change: The folder path with the results of the cleaning versions
        # pairwise_eval_folderpath_base: The folder path with the results of the baseline (no cleaning) version
        #
        # Notice that task_metrics in this case is for display purposes only, since there is only one metric here in the data.
        # It can be something like ["pairwise_ranking"], and in any case is mandatory.
        
        super().__init__(base_dir, data_folder_names_ordered, task_name, task_model_names, 
                         task_metrics, output_folder, 
                         cleaning_versions_to_show, audio_metrics, output_figures)
        self.pairwise_eval_folderpath_change = pairwise_eval_folderpath_change
        self.pairwise_eval_folderpath_base = pairwise_eval_folderpath_base

    def _get_pairwise_results(self, pairwise_eval_folderpath):
        # The task results are retrieved differently than in the base class.

        # for the pairwise results json files:
        system_name_to_pairwise_scores = {}  # system name -> cleaning version name -> pairwise comparison score per noise leveland clean_version_id is the type of cleaning done to the transcript (e.g. cleaned_noun, or '' if none)
        for filename in os.listdir(pairwise_eval_folderpath):
            if filename.startswith('results') and filename.endswith('json'):
                version_name = filename.replace('results_', '')[:-5]  # e.g. results_Gpt4oMini_truncate_cleaned_adj.json -> Gpt4oMini_truncate_cleaned_adj
                if '_cleaned_' in version_name:
                    clean_ver_id_char_idx = version_name.index('_cleaned_')
                    clean_ver_id = version_name[clean_ver_id_char_idx + 1:]  # e.g. "Gpt4oMini_truncate_cleaned_adj" -> "cleaned_adj"
                    system_version_name = version_name[:clean_ver_id_char_idx]  # e.g. "Gpt4oMini_truncate_cleaned_adj" -> "Gpt4oMini_truncate"
                else:
                    clean_ver_id = ''
                    system_version_name = version_name

                system_pairwise_eval_filepath = os.path.join(pairwise_eval_folderpath, filename)
                if system_version_name not in system_name_to_pairwise_scores:
                    system_name_to_pairwise_scores[system_version_name] = {}
                with open(system_pairwise_eval_filepath, 'r', encoding='utf-8', errors='ignore') as fIn:
                    system_pairwise_eval_results = json.load(fIn)['overall']['all']
                system_name_to_pairwise_scores[system_version_name][clean_ver_id] = system_pairwise_eval_results

        return system_name_to_pairwise_scores


    def _get_cwer_scores_for_noise_level(self, transcripts_results_data_folder_name):
        clean_ver_to_cwer_scores = {}  # clean_version_id -> metric -> (<mean>, <std>)   where clean_version_id is the type of cleaning done to the transcript (e.g. cleaned_noun)

        # a the transcription jsonl files:
        if transcripts_results_data_folder_name != '':
            output_cwer_filepath_template = os.path.join(self.base_dir, 'results',transcripts_results_data_folder_name, 'cwer_resultsCLEANVERSIONID.json')
            transcripts_folder_path = os.path.join(self.base_dir, 'transcripts', transcripts_results_data_folder_name)
            for filename in tqdm(os.listdir(transcripts_folder_path)):
                if filename.startswith('whisper_small') and filename.endswith('.jsonl'):
                    input_transcription_filepath = os.path.join(transcripts_folder_path, filename)
                    if '_cleaned_' in filename:
                        clean_ver_id_char_idx = filename.index('_cleaned_')
                        clean_ver_id = filename[clean_ver_id_char_idx: -6]  # e.g. "whisper_small__qmsum_clean_test_cleaned_adj.jsonl" -> "_cleaned_adj"
                    else:
                        clean_ver_id = ''
                    output_cwer_filepath = output_cwer_filepath_template.replace('CLEANVERSIONID', clean_ver_id)
                    cwer_overall = self._get_cwer_scores(input_transcription_filepath, output_cwer_filepath)
                    clean_ver_to_cwer_scores[clean_ver_id[1:]] = cwer_overall  # (remove trailing '_' from e.g. _cleaned_adj)

        return clean_ver_to_cwer_scores


    def _prepare_scores_for_display(self):
        # This function is overriden against the base class since the results are loaded differently for the pairwise ranking.

        self.noise_name_to_clean_ver_to_cwer_scores = {}
        self.noise_names_ordered = []
        for data_folder_name, noise_name in self.data_folder_names_ordered:
            print(data_folder_name)
            clean_ver_to_cwer_scores = self._get_cwer_scores_for_noise_level(data_folder_name)
            self.noise_name_to_clean_ver_to_cwer_scores[noise_name] = clean_ver_to_cwer_scores
            self.noise_names_ordered.append(noise_name)

        print(f'pairwise results from: {self.pairwise_eval_folderpath_base} for base and {self.pairwise_eval_folderpath_change} for change')
        self.task_model_to_pairwise_results_change = self._get_pairwise_results(self.pairwise_eval_folderpath_change)
        self.task_model_to_pairwise_results_base = self._get_pairwise_results(self.pairwise_eval_folderpath_base)


    def _get_xy_from_scores_no_clean(self, task_model_name, task_metric, audio_metric):
        # This function is overriden against the base class since the results are read differently for the pairwise ranking,
        # e.g., from two different results files (for baseline and cleaned versions), and adding 1 to the y-value (see explanation below)
        # Notice that the task_metric is not actually used here.

        x_list = []
        y_list = []
        y_err_lower_list = []
        y_err_upper_list = []
        for noise_name in self.noise_names_ordered:
            # The QMSum pairwise comparison is in the format: task_model_name -> cleaning_method -> noise_name -> <scores info>
            # The addition of 1 to the y-value is because the pairwise comparison process for the baseline (no cleaning) verion
            #   did not compare a summary against itself, in which case it would be a tie. But for the cleaned versions, a cleaned
            #   summary is compared against all baseline summaries, including its uncleaned counterpart, so it has potential for more points.
            y_list.append(self.task_model_to_pairwise_results_base[task_model_name][''][noise_name]['mean'] + 1)
            y_err_lower_list.append(self.task_model_to_pairwise_results_base[task_model_name][''][noise_name]['confidence_interval_95'][0] + 1)
            y_err_upper_list.append(self.task_model_to_pairwise_results_base[task_model_name][''][noise_name]['confidence_interval_95'][1] + 1)

            # the x-values are the WER/audio score:
            if noise_name != 'original':
                x_list.append(self.noise_name_to_clean_ver_to_cwer_scores[noise_name][''][audio_metric][0])
            else:  # for the original transcript (no noise at all), there is no cleaning_version, and the WER is always zero
                x_list.append(0)

        return x_list, y_list, y_err_lower_list, y_err_upper_list
    

    def _get_xy_from_scores(self, task_model_name, cleaning_version, task_metric, audio_metric):
        # This function is overriden against the base class since the results are read differently for the pairwise ranking,
        # e.g., from two different results files (for baseline and cleaned versions), and adding 1 to the y-value (see explanation below)
        # Notice that the task_metric is not actually used here.

        x_list = []
        y_list = []
        y_err_lower_list = []
        y_err_upper_list = []
        for noise_name in self.noise_names_ordered:
            # The QMSum pairwise comparison is in the format: task_model_name -> cleaning_method -> noise_name -> <scores info>
            # The addition of 1 to the y-value is because the pairwise comparison process for the baseline (no cleaning) verion
            #   did not compare a summary against itself, in which case it would be a tie (socre of 1). But for the cleaned versions,
            #   a cleaned summary is compared against all baseline summaries, including its uncleaned counterpart, so it has 
            #   potential for more points.
            # The '{noise_name}__new' noise version is used, because this is the name for the cleaned versions, when comparing
            #   against the baseline version. I.e., each summary in the cleaned version is compared against the respective summaries 
            #   of the baseline version.
            if noise_name != 'original':
                x_list.append(self.noise_name_to_clean_ver_to_cwer_scores[noise_name][cleaning_version][audio_metric][0])
                y_list.append(self.task_model_to_pairwise_results_change[task_model_name][cleaning_version][f'{noise_name}__new']['mean'])
                y_err_lower_list.append(self.task_model_to_pairwise_results_change[task_model_name][cleaning_version][f'{noise_name}__new']['confidence_interval_95'][0] )
                y_err_upper_list.append(self.task_model_to_pairwise_results_change[task_model_name][cleaning_version][f'{noise_name}__new']['confidence_interval_95'][1])
            else:
                x_list.append(0)
                y_list.append(self.task_model_to_pairwise_results_base[task_model_name][''][noise_name]['mean'] + 1)
                y_err_lower_list.append(self.task_model_to_pairwise_results_base[task_model_name][''][noise_name]['confidence_interval_95'][0] + 1)
                y_err_upper_list.append(self.task_model_to_pairwise_results_base[task_model_name][''][noise_name]['confidence_interval_95'][1] + 1)

        return  x_list, y_list, y_err_lower_list, y_err_upper_list
    



class LatexTableGenerator:

    def map_names_in_baselines_data(self, curve_scores_baselines_all, name_mappings):
        """
        Replace all names in the data dictionary based on the given mappings.

        Parameters:
        - data: The original data dictionary.
        - name_mappings: Dictionary with mappings for models, tasks, and metrics.

        Returns:
        - A new dictionary with updated names.
        """
        mapped_data = {}

        # Map model names
        for model, tasks in curve_scores_baselines_all.items():
            new_model = name_mappings.get(model, model)
            mapped_data[new_model] = {}

            # Map task names
            for task, metrics in tasks.items():
                new_task = name_mappings.get(task, task)
                mapped_data[new_model][new_task] = {}

                # Map metric names
                for metric, values in metrics.items():
                    new_metric = name_mappings.get(metric, metric)
                    mapped_data[new_model][new_task][new_metric] = values

        return mapped_data


    def create_latex_table_for_baselines_curve_scores(self, curve_scores_baselines_all, name_mappings):
        """
        Generate a LaTeX table from a nested dictionary of the form:
        model_name -> task_name -> task_metric -> {'auc': <value>, 'auc_interval': <value>, 'ntp': <value>}
        """
        curve_scores_baselines_all = self.map_names_in_baselines_data(curve_scores_baselines_all, name_mappings)

        # Extract unique model names and task names
        model_names = list(curve_scores_baselines_all.keys())
        task_names = set()
        for model in curve_scores_baselines_all.values():
            task_names.update(model.keys())
        #task_names = sorted(task_names)  # Ensure consistent order

        # Build the LaTeX table
        latex = []
        latex.append("\\begin{table*}[ht]")
        latex.append("    \\centering")
        latex.append("    \\resizebox{\\textwidth}{!}{%")
        
        # Start the tabular environment
        col_count = len(model_names) * 2 + 2
        col_format = "c|l" + "|cc" * len(model_names)
        latex.append(f"    \\begin{{tabular}}{{{col_format}}}")
        latex.append("    \\toprule")
        
        # Header rows
        model_header = " & & " + " & ".join([f"\\multicolumn{{2}}{{c|}}{{{model}}}" for model in model_names]) + " \\\\"
        latex.append(model_header)
        latex.append("    & Metric & " + " & ".join(["AUC & NTP"] * len(model_names)) + " \\\\")
        latex.append("    \\midrule")
        
        # Fill the table rows
        for task in task_names:
            task_metrics = set()
            for model in curve_scores_baselines_all.values():
                task_metrics.update(model.get(task, {}).keys())
            #task_metrics = sorted(task_metrics)  # Ensure consistent order
            
            # Add task name row (spanning multiple metrics)
            latex.append(f"    \\small{{\\multirow{{{len(task_metrics)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{task}}}}}}}")
            
            for i, metric in enumerate(task_metrics):
                if i > 0:  # Add metric row without task name
                    latex.append("    &")
                else:
                    latex[-1] += " &"

                # Collect AUC values for the metric
                auc_values = [
                    round(curve_scores_baselines_all[model].get(task, {}).get(metric, {}).get("auc", float("-inf")), 3)
                    for model in model_names
                ]
                max_auc = max(auc_values)  # Find the maximum AUC
                
                # Fill metric name and scores
                metric_row = [f"    {metric}"]
                for auc, model in zip(auc_values, model_names):
                    scores = curve_scores_baselines_all[model].get(task, {}).get(metric, {})
                    auc_str = f"\\textbf{{{auc:.3f}}}" if auc == max_auc else f"{auc:.3f}"
                    auc_interval_str = f"{scores.get('auc_interval', 0):.3f}"
                    #ntp_str = f"{scores.get('ntp', ['-'])[0]:.3f}"
                    ntp_str = f"{scores['ntp'][0]:.3f}" if scores['ntp'] else '-'
                    metric_row.append(f"    {auc_str} \\small{{± {auc_interval_str}}} & {ntp_str}")
                latex.append(" & ".join(metric_row) + " \\\\")

            latex.append("    \\midrule")
        
        if 'midrule' in latex[-1]:
            latex = latex[:-1]
        latex.append("    \\bottomrule")
        latex.append("    \\end{tabular}%")
        latex.append("    }")
        latex.append("    \\caption{Model Performance Scores}")
        latex.append("    \\label{tab_scores_baselines}")
        latex.append("\\end{table*}")
        
        return "\n".join(latex)



    def map_names_in_cleaning_data(self, curve_scores_cleaning_all, name_mappings):
        """
        Replace all names in the data dictionary based on the given mappings.

        Parameters:
        - data: The original data dictionary.
        - name_mappings: Dictionary with mappings for models, tasks, metrics, and cleaning methods.

        Returns:
        - A new dictionary with updated names.
        """
        mapped_data = {}

        # Map model names
        for model, tasks in curve_scores_cleaning_all.items():
            new_model = name_mappings.get(model, model)
            mapped_data[new_model] = {}

            # Map task names
            for task, metrics in tasks.items():
                new_task = name_mappings.get(task, task)
                mapped_data[new_model][new_task] = {}

                # Map metric names
                for metric, cleaning in metrics.items():
                    new_metric = name_mappings.get(metric, metric)
                    mapped_data[new_model][new_task][new_metric] = {}

                    for clean_name, values in cleaning.items():
                        new_clean_name = name_mappings.get(clean_name, clean_name)
                        mapped_data[new_model][new_task][new_metric][new_clean_name] = values

        return mapped_data


    def create_latex_table_for_cleaning_curve_scores(self, curve_scores_cleaning_all, name_mappings):
        """
        Generate a LaTeX table from the given data structure.

        Parameters:
        - data: Dictionary with the structure
        task_model_name -> task_name -> task_metric -> {'ces': <value>}
        
        Returns:
        - A string containing the LaTeX table.
        """
        curve_scores_cleaning_all = self.map_names_in_cleaning_data(curve_scores_cleaning_all, name_mappings)

        # Extract task and model names
        model_names = list(curve_scores_baselines_all.keys())

        # Group the data by task and metric
        task_metrics = {}
        for model, tasks in curve_scores_cleaning_all.items():
            for task_name, metrics in tasks.items():
                if task_name not in task_metrics:
                    task_metrics[task_name] = {}
                for metric_name, values in metrics.items():
                    if metric_name not in task_metrics[task_name]:
                        task_metrics[task_name][metric_name] = {}
                    task_metrics[task_name][metric_name][model] = values

        # Start building the LaTeX table
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("    \\centering")
        latex.append("    \\resizebox{\\columnwidth}{!}{%")
        col_format = "c|l" + "|".join(["c"] * len(model_names))
        latex.append(f"    \\begin{{tabular}}{{{col_format}}}")
        latex.append("    \\toprule")
        
        # Header row
        header_row = " & & " + " & ".join(model_names) + " \\\\"
        latex.append(header_row)
        latex.append("    \\midrule")
        

        # Fill rows for each task and metric
        for task_index, (task_name, metrics) in enumerate(task_metrics.items()):
            if task_index > 0:
                latex.append("    \\midrule")  # Add midrule between tasks

            for metric_index, (metric_name, model_values) in enumerate(metrics.items()):
                if metric_index > 0:
                    latex.append(f"    \\cmidrule(lr){{2-{len(model_names) + 2}}}")  # Add cmidrule between metrics

                cleaning_versions = list(set(cv for values in model_values.values() for cv in values.keys()))
                cleaning_versions.sort()



                # In this version, the scores of the cleaning techniques for each metric and model are colored according to their ranking.
                # START HERE RANKING COLORING

                # Extract values for ranking and normalize them
                column_ranks = {model: [] for model in model_names}
                for model in model_names:
                    column_ranks[model] = [
                        model_values.get(model, {}).get(cv, {}).get("ces", float('-inf')) for cv in cleaning_versions
                    ]
                
                # Calculate ranks and map to grayscale
                column_colors = {}
                for model, values in column_ranks.items():

                    # Rank values based on original list, handle ties
                    ranked_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
                    ranked_values = [values[i] for i in ranked_indices]
                    
                    ranks = [0] * len(values)
                    current_rank = 0
                    for i in range(1, len(ranked_values)):
                        if ranked_values[i] == ranked_values[i-1]:
                            ranks[ranked_indices[i]] = current_rank
                        else:
                            current_rank = i
                            ranks[ranked_indices[i]] = current_rank

                    # Normalize ranks to grayscale, lower rank (higher value) -> lighter color
                    column_colors[model] = [1 - (rank / (len(values) - 1)) if len(values) > 1 else 1 for rank in ranks]
                
                # Add task and metric with row span
                row_span = len(cleaning_versions)
                latex.append(f"    \\multirow{{{row_span}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{task_name} | {metric_name}}}}}")
                
                # Add rows for each cleaning version
                for i, cv in enumerate(cleaning_versions):
                    latex.append("    &")  # Indent for subsequent rows
                    row = [f"    {cv}"]
                    for model in model_names:
                        value = model_values.get(model, {}).get(cv, {}).get("ces", "-")
                        color = f"{column_colors[model][i]:.2f}" if isinstance(value, (int, float)) else "1.0"
                        text_color = "black" if column_colors[model][i] > 0.5 else "white"  # Adjust text color based on background color
                        formatted_value = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
                        row.append(f"    \\cellcolor[gray]{{{color}}} \\textcolor{{{text_color}}}{{{formatted_value}}}")
                    latex.append(" & ".join(row) + " \\\\")

                # END HERE RANKING COLORING


                # In this version, the max score of the coloring techniques is emboldened for each metric and model
                # START HERE BOLD MAX VAL

                # # Find maximum values for bolding
                # max_values = {}
                # for model in model_names:
                #     max_values[model] = max(
                #         (model_values.get(model, {}).get(cv, {}).get("ces", float('-inf')) for cv in cleaning_versions),
                #         default=float('-inf')
                #     )

                # # Add task and metric with row span
                # row_span = len(cleaning_versions)
                # latex.append(f"    \\multirow{{{row_span}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{task_name} | {metric_name}}}}}")
                
                # # Add rows for each cleaning version
                # for i, cv in enumerate(cleaning_versions):
                #     #if i > 0:
                #     latex.append("    &")  # Indent for subsequent rows
                #     row = [f"    {cv}"]
                #     for model in model_names:
                #         value = model_values.get(model, {}).get(cv, {}).get("ces", "-")
                #         if isinstance(value, (int, float)) and value == max_values[model]:
                #             row.append(f"    \\textbf{{{value:.3f}}}")
                #         else:
                #             row.append(f"    {value:.3f}" if isinstance(value, (int, float)) else str(value))
                #     latex.append(" & ".join(row) + " \\\\")

                # END HERE BOLD MAX VAL
            
        if 'midrule' in latex[-1]:
            latex = latex[:-1]
        latex.append("    \\bottomrule")
        latex.append("    \\end{tabular}%")
        latex.append("    }")
        latex.append("    \\caption{Model Performance Scores}")
        latex.append("    \\label{tab_scores_cleaning}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)




def main_mrda(base_dir, cleaning_versions_to_show, audio_metrics, task_model_names, output_folder, output_figures=True):
    # data_folder_paths_ordered = [
    #     (f'{base_dir}/transcripts/mrda_test___reverb__noise_-10', '-10'),
    #     (f'{base_dir}/transcripts/mrda_test___reverb__noise_-5', '-5'),
    #     (f'{base_dir}/transcripts/mrda_test___reverb__noise_0', '0'),
    #     (f'{base_dir}/transcripts/mrda_test___reverb__noise_5', '5'),
    #     (f'{base_dir}/transcripts/mrda_test___reverb__noise_10', '10'),
    #     (f'{base_dir}/transcripts/mrda_test', 'none'),
    #     (f'{base_dir}/transcripts/mrda_test_source', 'original')
    # ]
    data_folder_names_ordered = [
        ('mrda_test___reverb__noise_-10', '-10'),
        ('mrda_test___reverb__noise_-5', '-5'),
        ('mrda_test___reverb__noise_0', '0'),
        ('mrda_test___reverb__noise_5', '5'),
        ('mrda_test___reverb__noise_10', '10'),
        ('mrda_test', 'none'),
        ('mrda_test_source', 'original')
    ]
    task_name = 'mrda'
    task_metrics = ['f1_macro', 'accuracy']

    results_displayer = ResultsDisplayer(base_dir, data_folder_names_ordered, task_name, task_model_names, task_metrics, output_folder,
                                         cleaning_versions_to_show, audio_metrics, output_figures)
    task_model_to_task_metric_to_curve_scores_baselines, task_model_to_task_metric_to_cleaning_version_to_curve_scores = \
        results_displayer.execute()
    return task_model_to_task_metric_to_curve_scores_baselines, task_model_to_task_metric_to_cleaning_version_to_curve_scores


def main_qaconv(base_dir, cleaning_versions_to_show, audio_metrics, task_model_names, output_folder, output_figures=True):
    # data_folder_paths_ordered = [
    #     (f'{base_dir}/transcripts/qaconv_test___reverb__noise_-10', '-10'),
    #     (f'{base_dir}/transcripts/qaconv_test___reverb__noise_-5', '-5'),
    #     (f'{base_dir}/transcripts/qaconv_test___reverb__noise_0', '0'),
    #     (f'{base_dir}/transcripts/qaconv_test___reverb__noise_5', '5'),
    #     (f'{base_dir}/transcripts/qaconv_test___reverb__noise_10', '10'),
    #     (f'{base_dir}/transcripts/qaconv_test', 'none'),
    #     (f'{base_dir}/transcripts/qaconv_test_source', 'original')
    # ]
    data_folder_names_ordered = [
        ('qaconv_test___reverb__noise_-10', '-10'),
        ('qaconv_test___reverb__noise_-5', '-5'),
        ('qaconv_test___reverb__noise_0', '0'),
        ('qaconv_test___reverb__noise_5', '5'),
        ('qaconv_test___reverb__noise_10', '10'),
        ('qaconv_test', 'none'),
        ('qaconv_test_source', 'original')
    ]
    task_name = 'qaconv'
    task_metrics = ['exact', 'f1', 'fzr']

    results_displayer = ResultsDisplayer(base_dir, data_folder_names_ordered, task_name, task_model_names, task_metrics, output_folder,
                                         cleaning_versions_to_show, audio_metrics, output_figures)
    task_model_to_task_metric_to_curve_scores_baselines, task_model_to_task_metric_to_cleaning_version_to_curve_scores = \
        results_displayer.execute()
    return task_model_to_task_metric_to_curve_scores_baselines, task_model_to_task_metric_to_cleaning_version_to_curve_scores


def main_qmsum(base_dir, cleaning_versions_to_show, audio_metrics, task_model_names, output_folder, output_figures=True):
    # the task model names are different for QMSum:
    task_model_names = ['Mistral7BInstruct_recursive', 'Llama3Instruct_recursive', 'Llama3_1Instruct_truncate', 'Gpt4oMini_truncate']
    task_name = 'qmsum'

    # for the ROUGE metrics:
    # data_folder_paths_ordered = [
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_-10', '-10'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_-5', '-5'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_0', '0'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_5', '5'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_10', '10'),
    #     (f'{base_dir}/transcripts/qmsum_test', 'none'),
    #     (f'{base_dir}/transcripts/qmsum_test_source', 'original')
    # ]
    data_folder_names_ordered = [
        ('qmsum_test___reverb__noise_-10', '-10'),
        ('qmsum_test___reverb__noise_-5', '-5'),
        ('qmsum_test___reverb__noise_0', '0'),
        ('qmsum_test___reverb__noise_5', '5'),
        ('qmsum_test___reverb__noise_10', '10'),
        ('qmsum_test', 'none'),
        ('qmsum_test_source', 'original')
    ]
    task_metrics = ['rouge1', 'rouge2', 'rougeL']
    results_displayer = ResultsDisplayer(base_dir, data_folder_names_ordered, task_name, task_model_names, task_metrics, output_folder,
                                         cleaning_versions_to_show, audio_metrics, output_figures)
    model_to_metric_to_curve_scores_baselines_standard, model_to_metric_to_cleaning_to_curve_scores_standard = \
        results_displayer.execute()
    
    # for the pairwise ranking metric:
    # data_folder_paths_ordered = [
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_-10', 'noise_m10'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_-5', 'noise_m5'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_0', 'noise_0'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_5', 'noise_5'),
    #     (f'{base_dir}/transcripts/qmsum_test___reverb__noise_10', 'noise_10'),
    #     (f'{base_dir}/transcripts/qmsum_test', 'no_noise'),
    #     ('', 'original')  # no path means there is no CWER for this version (because there is no WER on the original gold transcript)
    # ]
    data_folder_names_ordered = [
        ('qmsum_test___reverb__noise_-10', 'noise_m10'),
        ('qmsum_test___reverb__noise_-5', 'noise_m5'),
        ('qmsum_test___reverb__noise_0', 'noise_0'),
        ('qmsum_test___reverb__noise_5', 'noise_5'),
        ('qmsum_test___reverb__noise_10', 'noise_10'),
        ('qmsum_test', 'no_noise'),
        ('', 'original')  # no path means there is no CWER for this version (because there is no WER on the original gold transcript)
    ]
    pairwise_eval_folderpath_change = f'{base_dir}/results/qmsum_test_pairwise_eval_baseline_change'
    pairwise_eval_folderpath_base = f'{base_dir}/results/qmsum_test_pairwise_eval'
    task_metrics = ['pairwise_ranking']  # no metric here, it's just a placeholder
    
    results_displayer = ResultsDisplayerPairwiseRanking(base_dir, data_folder_names_ordered, 
                                                        task_name, task_model_names, task_metrics, 
                                                        output_folder,
                                                        pairwise_eval_folderpath_change, 
                                                        pairwise_eval_folderpath_base, 
                                                        cleaning_versions_to_show, audio_metrics, output_figures)
    model_to_metric_to_curve_scores_baselines_pairwise, model_to_metric_to_cleaning_to_curve_scores_pairwise = \
        results_displayer.execute()
    
    # merge the curve scores dictionaries, viewing pairwise as another metric:
    model_to_metric_to_curve_scores_baselines_all = {}
    for task_model in model_to_metric_to_curve_scores_baselines_standard:
        model_to_metric_to_curve_scores_baselines_all[task_model] = \
            model_to_metric_to_curve_scores_baselines_standard[task_model] | model_to_metric_to_curve_scores_baselines_pairwise[task_model]
    model_to_metric_to_cleaning_to_curve_scores_all = {}
    for task_model in model_to_metric_to_cleaning_to_curve_scores_standard:
        model_to_metric_to_cleaning_to_curve_scores_all[task_model] = \
            model_to_metric_to_cleaning_to_curve_scores_standard[task_model] | model_to_metric_to_cleaning_to_curve_scores_pairwise[task_model]
    
    return model_to_metric_to_curve_scores_baselines_all, model_to_metric_to_cleaning_to_curve_scores_all
    

def merge_curve_scores_on_task_model(task_name_to_scores, name_mapping):
    curve_scores_merged = {name_mapping[model_name]: {} for model_name in list(task_name_to_scores.values())[0].keys()}
        
    for task_name in task_name_to_scores:
        for model_name in task_name_to_scores[task_name]:
            task_name_mapped = name_mapping[task_name]
            model_name_mapped = name_mapping[model_name]
            curve_scores_merged[model_name_mapped][task_name_mapped] = task_name_to_scores[task_name][model_name]

    return curve_scores_merged  # model_name -> task_name -> task_metric -> <scores>
                                #   baselines: <scores> =   {'auc', 'auc_interval', 'ntp'}
                                #   cleaning:  <scores> =   cleaning_method -> {'ces'}


if __name__ == '__main__':
    # usage:
    # python3 run_assessment_on_results.py <tasks>
    # tasks: names of tasks space separated (qmsum, qaconv, mrda)
    #
    # Set the cleaning_versions_to_show for the ckeaning techniques to show on the graph
    # and the output_folder in which to dump the figures.
    BASE_DIR = config.BASE_PATH

    if len(sys.argv) > 1:
        tasks_to_process = sys.argv[1:]
    else:
        tasks_to_process = ['qmsum', 'qaconv', 'mrda']

    cleaning_versions_to_show = ['', 'cleaned_noun', 'cleaned_verb', 'cleaned_adj', 'cleaned_adv', 'cleaned_content', 'cleaned_noncontent', 'cleaned_named_entity'] # None means all of them ['', 'cleaned_noun', 'cleaned_verb', 'cleaned_adj', 'cleaned_adv', 'cleaned_noncontent', 'cleaned_content', 'cleaned_named_entity']
    output_folder = f'{BASE_DIR}/results/figures/all'
    
    # if we only need part of the cleaning techniques on the graphs, then use this instead of the above:
    #cleaning_versions_to_show = ['', 'cleaned_noun', 'cleaned_adv', 'cleaned_content', 'cleaned_noncontent', 'cleaned_named_entity']
    #output_folder = f'{BASE_DIR}/results/figures/partial'

    audio_metrics = ['wer'] # 'cer'
    task_model_names = ['Mistral7BInstruct', 'Llama3Instruct', 'Llama3_1Instruct', 'Gpt4oMini']
    output_figures = True
    
    os.makedirs(output_folder, exist_ok=True)

    name_mappings_for_tables = {
        "Mistral7BInstruct": "Mistral",
        "Mistral7BInstruct_recursive": "Mistral",
        "Llama3Instruct": "Llama3",
        "Llama3Instruct_recursive": "Llama3",
        "Llama3_1Instruct": "Llama3.1",
        "Llama3_1Instruct_truncate": "Llama3.1",
        "Gpt4oMini": "GPT4oMini",
        "Gpt4oMini_truncate": "GPT4oMini",
        "qmsum": "QMSum",
        "qaconv": "QAConv",
        "mrda": "MRDA",
        "rouge1": 'R-1',
        "rouge2": 'R-2',
        "rougeL": 'R-L',
        "pairwise_ranking": "PW-Rank",
        "f1": "$F_1$",
        "exact": "Exact",
        "fzr": "Fuzzy",
        "f1_macro": "Mac-$F_1$",
        "accuracy": "Acc",
        "cleaned_adj": "Adjectives",
        "cleaned_adv": "Adverbs",
        "cleaned_content": "Content",
        "cleaned_named_entity": "Named-ents",
        "cleaned_noncontent": "Non-content",
        "cleaned_noun": "Nouns",
        "cleaned_verb": "Verbs",
    }

    task_to_scores_baselines = {}
    task_to_scores_cleaning = {}
    if 'mrda' in tasks_to_process:
        curve_scores_baselines_mrda, curve_scores_cleaning_mrda = \
            main_mrda(BASE_DIR, cleaning_versions_to_show, audio_metrics, 
                    task_model_names, output_folder, output_figures)
        task_to_scores_baselines['mrda'] = curve_scores_baselines_mrda
        task_to_scores_cleaning['mrda'] = curve_scores_cleaning_mrda
    if 'qaconv' in tasks_to_process:
        curve_scores_baselines_qaconv, curve_scores_cleaning_qaconv = \
            main_qaconv(BASE_DIR, cleaning_versions_to_show, audio_metrics, 
                        task_model_names, output_folder, output_figures)
        task_to_scores_baselines['qaconv'] = curve_scores_baselines_qaconv
        task_to_scores_cleaning['qaconv'] = curve_scores_cleaning_qaconv
    if 'qmsum' in tasks_to_process:
        curve_scores_baselines_qmsum, curve_scores_cleaning_qmsum = \
            main_qmsum(BASE_DIR, cleaning_versions_to_show, audio_metrics, 
                    task_model_names, output_folder, output_figures)
        task_to_scores_baselines['qmsum'] = curve_scores_baselines_qmsum
        task_to_scores_cleaning['qmsum'] = curve_scores_cleaning_qmsum

    curve_scores_baselines_all = \
        merge_curve_scores_on_task_model(task_to_scores_baselines, name_mappings_for_tables)
    curve_scores_cleaning_all = \
        merge_curve_scores_on_task_model(task_to_scores_cleaning, name_mappings_for_tables)
    
    
    # create the Latex tables with the scores:
    latex_table_generator = LatexTableGenerator()
    latex_table_baselines = latex_table_generator.create_latex_table_for_baselines_curve_scores(curve_scores_baselines_all, name_mappings_for_tables)
    latex_table_cleaning = latex_table_generator.create_latex_table_for_cleaning_curve_scores(curve_scores_cleaning_all, name_mappings_for_tables)
    with open(os.path.join(output_folder, 'table_baseline_scores.tex'), 'w') as fOut:
        fOut.write(latex_table_baselines)
    with open(os.path.join(output_folder, 'table_cleaning_scores.tex'), 'w') as fOut:
        fOut.write(latex_table_cleaning)

'''
To get all the results for all tasks, models and eval metrics, run:
python3 assess_results_all.py
from the audio folder.
The graphs and tables will be dumped into a folder, as set in the main function, e.g., "figures".
'''