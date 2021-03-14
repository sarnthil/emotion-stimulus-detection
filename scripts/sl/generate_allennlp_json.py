import sys
import json
import click


@click.command()
@click.option("--train")
@click.option("--dev")
def cli(train, dev):
    data = {
        "dataset_reader": {
            "type": "sl_reader",
            "token_indexers": {
                "tokens": {
                    "type": "single_id",
                    "namespace": "tokens",
                    "lowercase_tokens": True,
                }
            },
            "lazy": False,
        },
        "iterator": {
            "type": "bucket",
            "sorting_keys": [["tokens", "num_tokens"]],
            "batch_size": 10,
        },
        "train_data_path": train,
        "validation_data_path": dev,
        "model": {
            "type": "sl_model",
            "text_field_embedder": {
                "token_embedders": {
                    "tokens": {
                        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                        "type": "embedding",
                        "embedding_dim": 300,
                        "trainable": true,
                    }
                }
            },
            "seq2seq_encoder": {
                "type": "self-attentive-lstm",
                "input_size": 300,
                "hidden_size": 100,
                "num_layers": 2,
                "bidirectional": true,
            },
        },
        "trainer": {
            "num_epochs": 50,
            "patience": 10,
            "grad_clipping": 5.0,
            "validation_metric": "-loss",
            "optimizer": {"type": "adam", "lr": 0.003},
        },
    }

    json.dump(data, sys.stdout)


if __name__ == "__main__":
    cli()
