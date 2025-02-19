# ENDow: Measuring the *E*ffect of Transcription *N*oise on *Dow*nstream NLU Tasks

Speech-to-text technology (ASR) generates text for given recorded speech. Naturally there are errors in the transcripts, and the amount and types of errors fluctuate between use-cases. When transcripts are then sent on to downstream NLP tasks (such as summarization), the errors propagate and cause unexpected mistakes in the outputs of the NLP task.

The purpose of this project is to support the assessment of how much the transcription noise affects downstream tasks. A framework prepares transcripts at different levels and types of noise, and provides graphs with scores that attempt to analytically explain the effects.

This is the code for the assessment framework, documented in the paper *"Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks"* (see citation below).

## Code Overview

The run_*.py files consist of the code for the pipeline of the framework. The order for running the pipeline is as follows:

1. `run_tts_on_data.py` - runs text to speech on the gold transcripts, outputting wav files for the utterances
    - Input: transcripts from `<base_dir>/data/<dataset>`
    - Output: audio files for the utterances in `<base_dir>/audio/<dataset>/<datum>/<utterance_id>.wav`
3. `run_noiser_on_audio.py` - adds reverberation and background sounds to the audio files from the last step
    - Input: audio files for the utterances in `<base_dir>/audio/<dataset>/<datum>/<utterance_id>.wav`
    - Output: audio files for utterances at `<base_dir>/audio/<dataset>___reverb__noise_*/<datum>/<utterance_id>.wav`
4. `run_stt_on_audio.py` - runs stt on the audio files, providing transcripts with different levels of noise
    - Input: audio files for utterances at `<base_dir>/audio/<dataset>*`
    - Output: transcribed utterances at `<base_dir>/transcripts/<dataset>*/<asr>__<dataset>*.jsonl`
5. `run_cleaning_on_transcripts.py` - cleans transcripts with different techniques
    - Input: transcripts in `<base_dir>/transcripts/<dataset>*/<asr>__<dataset>*.jsonl`
    - Output: transcripts in `<base_dir>/transcripts/<dataset>/<asr>__<dataset>_*_cleaned_*.jsonl`
6. `run_inference_<dataset>.py` - runs inference with different models on all the transcripts
    - Input: transcripts in `<base_dir>/transcripts/<dataset>`
    - Output: outputs in `<base_dir>/results/<dataset>*/results_<model>*.json`
7. FOR SUMMARIZATION (QMSUM): run_pairwise_eval_qmsum.ipynb - runs pairwise comparison between output summaries
    - Input: outputs in `<base_dir>/results/qmsum_test*/results_<model>*.json`
    - Output: score results in `<base_dir>/results/qmsum_test_pairwise*`
8. run_assessment_on_results.py - evaluates results and computes WER scores for transcripts, and outputs results
    - Input: outputs in `<base_dir>/results/<dataset>*`
    - Output: tables and graphs in `<base_dir>/results/figures`

## Setup Requirements

- Operating system: The code was verified and run on a Linux server.
- Hardware: Some of the models in the pipeline require a GPU with up to about 20GB memory. The code was verified and run with an A100 server with 40GB memory.
- Python environment: Setup an environment (verified with Python 3.10) and install the requirements in the requirements.txt file.
- Configuration: Set the values in the config.py file.
    - BASE_PATH: the location where this directory is placed
    - HUGGINGFACE_ACCESS_TOKEN: A HuggingFace key for running some of the models used
    - AZURE_OPENAI_API_KEY: An OpenAI key for running on the Azure OpenAI API. If this is not possible for you, you can change the functionality in the generation_models.py code to use a different API for OpenAI.
- In data/noise_files place some background sound audio WAV files that can be used to add in the noised audio files. They can be any length, but 1 minute of audio is enough. Place several files to allow for randomness of background sounds in the noised audio files.
- In data/QAConv, place the article_full.json file from [here](https://github.com/salesforce/QAConv/blob/master/dataset/QAConv-V1.1.zip).

## Data

For our experiments, we used three existing SLU dataset, which are placed in the data folder here:
- [QMSum](https://github.com/Yale-LILY/QMSum/tree/main/data/ALL/test) for query-focused summarization
- [QAConv](https://github.com/salesforce/QAConv/blob/master/dataset/QAConv-V1.1.zip) for extractive question-answering
- [MRDA](https://github.com/NathanDuran/MRDA-Corpus/tree/master/mrda_data/test) for dialogue-act classification



## Changing the Framework

Any of the models and methods used can be changed. We use:
- Datasets/tasks: QMSum for summarization, QAConv for QA, MRDA for dialog-act classification
- TTS model: [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
- Noising: reverberation + background sounds at different SNRs
- STT model: [whisper-small](https://huggingface.co/openai/whisper-small)
- Transcript cleaning techniques: part-of-speech and named-entites
- Task models: [Mistral7B-instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), [Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Llama3.1-8B-instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [GPT-4o-Mini](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=global-standard%2Cstandard-chat-completions#gpt-4o-and-gpt-4-turbo)
- Task evaluation: ROUGE/pairwise for QMSum, fuzzy/exact/f1 for QAConv, accuracy/macro-f1 for MRDA


## Citation

If the code or paper is used, please cite the paper:
