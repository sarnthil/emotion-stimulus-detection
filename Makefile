all: workdata

clean:
	rm -rf workdata/*

workdata:
	mkdir workdata

# extracting datasets
workdata/extracted.json: scripts/extract.py
	python3 scripts/extract.py >workdata/extracted.json

# tokenize datasets
workdata/tokenized.json: scripts/retokenize.py workdata/extracted.json
	python3 scripts/retokenize.py <workdata/extracted.json >workdata/tokenized.json

# split datasets and select instances for manual annotation
workdata/splitted.json: scripts/split.py workdata/tokenized.json
	python3 scripts/split.py workdata/tokenized.json >workdata/splitted.json

# extract clauses from all datasets
workdata/clausified.json: scripts/clausify.py workdata/splitted.json
	python3 scripts/clausify.py workdata/splitted.json >workdata/clausified.json

# make sl data
workdata/sl: scripts/make_data_sl.py
	python3 scripts/make_data_sl.py

# make sl experiment jsons
workdata/sl/experiments: scripts/sl/generate_allennlp_json.py workdata/sl
	mkdir workdata/sl/experiments
	python3 scripts/sl/generate_allennlp_json.py --train workdata/sl/reman.train --dev workdata/sl/reman.dev > workdata/sl/experiments/reman.json
	python3 scripts/sl/generate_allennlp_json.py --train workdata/sl/eca.train --dev workdata/sl/eca.dev >workdata/sl/experiments/eca.json
	python3 scripts/sl/generate_allennlp_json.py --train workdata/sl/gne.train --dev workdata/sl/gne.train >workdata/sl/experiments/gne.json
	python3 scripts/sl/generate_allennlp_json.py --train workdata/sl/emotion-stimulus.train --dev workdata/sl/emotion-stimulus.dev >workdata/sl/experiments/emotion-stimulus.json
	python3 scripts/sl/generate_allennlp_json.py --train workdata/sl/electoral_tweets.train --dev workdata/sl/electoral_tweets.dev >workdata/sl/experiments/electoral_tweets.json

# train models and write them in allennlp-models/sl
workdata/allennlp-models/sl: workdata/sl/experiments
	mkdir workdata/allennlp-models/sl/
	allennlp train workdata/sl/experiments/eca.json -s workdata/allennlp-models/sl/eca --include-package scripts.sl.reader --include-package scripts.sl.lstm_attention_tagger
	allennlp train workdata/sl/experiments/emotion-stimulus.json -s workdata/allennlp-models/sl/emotion-stimulus --include-package scripts.sl.reader --include-package scripts.sl.lstm_attention_tagger
	allennlp train workdata/sl/experiments/electoral_tweets.json -s workdata/allennlp-models/sl/electoral_tweets --include-package scripts.sl.reader --include-package scripts.sl.lstm_attention_tagger
	allennlp train workdata/sl/experiments/gne.json -s workdata/allennlp-models/sl/gne --include-package scripts.sl.reader --include-package scripts.sl.lstm_attention_tagger
	allennlp train workdata/sl/experiments/reman.json -s workdata/allennlp-models/sl/reman --include-package scripts.sl.reader --include-package scripts.sl.lstm_attention_tagger

# make predictions based on the model
# depend on workdata/allennlp-models/sl, produce ...

# evaluate model and write the results in ...
# depend on predictions (which we first align)
