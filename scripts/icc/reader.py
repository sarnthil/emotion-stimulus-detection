from typing import Dict, List, Union
import logging
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

logger = logging.getLogger(__name__)


class DummyTokenizer(Tokenizer):
    def tokenize(self, text):
        return [Token(text=word) for word in text.split(" ")]


@DatasetReader.register("icc_reader")
class ICCDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = DummyTokenizer()  # assumes our tokens unchanged
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path)) as data_file:
            for line in data_file:
                try:
                    text, label = line.strip().split("\t")
                except ValueError as e:
                    continue
                instance = self.text_to_instance(text=text, label=label)
                if instance is not None:
                    yield instance

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[: self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(
        self, text: str, label: Union[str, int] = None
    ) -> Instance:
        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens, self._token_indexers))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)
