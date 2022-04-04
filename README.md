# Better-Path
This is an extension to the the paper *Predicting ConceptNet Path Quality Using Crowdsourced Assessments of Naturalness* by Yilun Zhou, Steven Schockaert, and Julie Shah

To run the project, one need to:

1. download the data from Zhou's repository via running the `download_data.sh` script under the `data` folder
2. download pre-trained features via running `download.sh` followed by `RUN_ME_FIRST.sh` in the `prepare_data` folder
3. extract additional features that we proposed from paths via running `python3 feature.py` under `code`
4. extract heuristic information that we proposed from paths via running `python3 heuristic.py` under `code`
5. You're all set, feel free to try out some of our experiments under `experiment.py`
