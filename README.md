# Sequence Labeling vs. Clause Classification for English Emotion Stimulus Detection
## Supplementary material: Code

This folder contains the code associated with our paper.

- `sources`: Contains the original datasets.
- `scripts`: Contains all of our programs to work with the data. We describe a
  few of these scripts in detail:
    - `extract.py`: Responsible for extracting the data from the original
      datasets and aggregates them. Further helped by `extract_*.py` files that
      handle the specific dataset.
    - `clausify.py`: This implements the algorithm shown in pseudocode in the
      paper: It splits text into clauses based on heuristics applied to a
      constituency parse tree.
    - `sl`/`icc`/`jcc`: Folders that contain similar structure and implement the
      models using AllenNLP. Each of the folders contains a dataset reader, a
      predictor, a templated model specification, and an alignment script for
      the predictions.
    - `eval`: Evaluation scripts can be found here.

In order to generate the aggregated dataset (assuming all data in `sources` is complete), run:

    make workdata/clausified.json

This requires the dependencies specified in `requirements.txt`.

Training a model can be done either via the Makefile:

    make workdata/allennlp-models/sl

, or directly through `allennlp train`, e.g.:

    allennlp train workdata/sl/experiments/gne.json -s workdata/allennlp-models/sl/gne --include-package scripts.sl.reader --include-package scripts.sl.lstm_attention_tagger

This requires the dependencies specified in `requirements.allennlp.txt`. Note
that the Makefile currently only trains the SL models; this will be finalized
upon acceptance such that the whole pipeline is fully reproducible.
