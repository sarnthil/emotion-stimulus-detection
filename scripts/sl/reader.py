from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

import itertools


def is_divider(line):
    return line.strip() == ""


@DatasetReader.register("sl_reader")
class SLDatasetReader(DatasetReader):
    def __init__(
        self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r") as conll_file:
            for divider, lines in itertools.groupby(conll_file, is_divider):
                if divider:
                    continue
                fields = [l.strip().split() for l in lines]
                fields = [l for l in zip(*fields)]
                tokens, ner_tags = fields

                yield self.text_to_instance(tokens, ner_tags)

    @overrides
    def text_to_instance(self, words: List[str], ner_tags: List[str]) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["tokens"] = tokens
        fields["label"] = SequenceLabelField(ner_tags, tokens)

        return Instance(fields)
