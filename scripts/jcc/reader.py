from typing import Dict, List
import json
from overrides import overrides
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    ListField,
    MetadataField,
    SequenceLabelField,
    TextField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("jcc_reader")
class JCCDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing sessions from the MHD study. Each line is expected to have
    the following format:

        {
            'id': '...',
            'clauses': [
                ['Tokens', 'in', 'first', 'clause', ...],
                ...,
                ['Tokens', 'in', 'last', 'clause', ...]
            ],
            'labels': ['...']
        }

    The output of ``read`` is a list of ``Instance``s with the fields:

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``. If this is ``True``, training will start
        sooner but take longer per batch. Use if dataset is too large to fit in
        memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer used to split the utterances into tokens. Defaults to
        ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{'tokens': SingleIdTokenIndexer()}``.
    """

    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super(JCCDatasetReader, self).__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                instance_json = json.loads(line)
                instance_id = instance_json["id"]
                clauses = instance_json["clauses"]
                labels = instance_json["labels"]
                yield self.text_to_instance(instance_id, clauses, labels)

    @overrides
    def text_to_instance(
        self, instance_id: str, clauses: List[List[str]], labels: List[str] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        # Since each instance consists of a sequence of clauses, which themselves are sequences
        # of words, we need to use a ``ListField`` to properly handle the nested structure.
        tokenized_clauses = [[Token(word) for word in clause] for clause in clauses]
        sequence = ListField(
            [TextField(x, self._token_indexers) for x in tokenized_clauses]
        )
        fields["clauses"] = sequence
        if labels is not None:
            fields["labels"] = SequenceLabelField(labels, sequence)
        fields["metadata"] = MetadataField(
            {
                "id": instance_id,
                "clauses_text": [
                    [x.text for x in clause] for clause in tokenized_clauses
                ],
            }
        )
        return Instance(fields)
