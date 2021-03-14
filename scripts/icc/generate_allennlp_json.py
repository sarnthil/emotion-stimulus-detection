import sys
import json
import click


@click.command()
@click.option("--train")
@click.option("--dev")
def cli(train, dev):
    data = {
        "dataset_reader": {"type": "icc_reader"},
        "train_data_path": train,
        "validation_data_path": dev,
        "model": {
            "type": "icc_model",
            "text_field_embedder": {
                "token_embedders": {
                    "tokens": {
                        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                        "type": "embedding",
                        "embedding_dim": 300,
                        "trainable": False,
                    }
                }
            },
            "embedding_dropout": 0.2,
            "seq2seq_encoder": {
                "type": "self_attentive_lstm",
                "input_size": 300,
                "hidden_size": 100,
                "num_layers": 2,
                "bidirectional": True,
            },
            "classifier_feedforward": {
                "input_dim": 200,
                "num_layers": 2,
                "hidden_dims": [100, 2],
                "activations": ["relu", "linear"],
                "dropout": [0.5, 0.0],
            },
        },
        "iterator": {
            "type": "bucket",
            "batch_size": 32,
            "sorting_keys": [["tokens", "num_tokens"]],
        },
        "trainer": {"optimizer": "adam", "num_epochs": 50, "patience": 10},
    }
    json.dump(data, sys.stdout)


if __name__ == "__main__":
    cli()
