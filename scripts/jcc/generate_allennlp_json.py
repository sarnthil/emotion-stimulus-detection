import sys
import json
import click


@click.command()
@click.option("--train")
@click.option("--dev")
def cli(train, dev):
    data = {
        "vocabulary": {"min_count": {"tokens": 5}},
        "dataset_reader": {
            "type": "jcc_reader",
            "token_indexers": {
                "tokens": {"type": "single_id", "lowercase_tokens": False}
            },
        },
        "train_data_path": train,
        "validation_data_path": dev,
        "model": {
            "type": "jcc_model",
            "dropout": 0.5,
            "text_field_embedder": {
                "tokens": {
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": True,
                }
            },
            "clauses_encoder": {
                "type": "lstm",
                "input_size": 300,
                "hidden_size": 150,
                "dropout": 0.2,
                "num_layers": 1,
                "bidirectional": True,
            },
            "outer_encoder": {
                "type": "self_attentive_lstm",
                "input_size": 300,
                "hidden_size": 150,
                "dropout": 0.2,
                "num_layers": 1,
                "bidirectional": True,
            },
        },
        "iterator": {
            "type": "bucket",
            "batch_size": 10,
            "sorting_keys": [["tokens", "num_tokens"]],
        },
        "trainer": {"optimizer": "adam", "num_epochs": 50, "patience": 10},
    }
    json.dump(data, sys.stdout)


if __name__ == "__main__":
    cli()
