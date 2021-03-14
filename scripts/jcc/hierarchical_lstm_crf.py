from typing import Dict, Optional, List, Any
from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.modules import (
    ConditionalRandomField,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TimeDistributed,
    TextFieldEmbedder,
)
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear


@Model.register("jcc_model")
class JCC(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        clauses_encoder: Seq2VecEncoder,
        outer_encoder: Seq2SeqEncoder,
        label_namespace: str = "labels",
        constraint_type: str = None,
        include_start_end_transitions: bool = True,
        dropout: float = None,
        loss_weights: Optional[List] = [],
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(JCC, self).__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.clauses_encoder = inner_encoder
        self.outer_encoder = outer_encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.label_projection_layer = TimeDistributed(
            Linear(outer_encoder.get_output_dim(), self.num_tags)
        )

        labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
        constraints = allowed_transitions(constraint_type, labels)
        self.crf = ConditionalRandomField(
            self.num_tags,
            constraints,
            include_start_end_transitions=include_start_end_transitions,
        )
        self.metrics = {"accuracy": Accuracy()}

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            clauses_encoder.get_input_dim(),
            "text field embedding dim",
            "clauses encoder input dim",
        )
        check_dimensions_match(
            clauses_encoder.get_output_dim(),
            outer_encoder.get_input_dim(),
            "clauses encoder output dim",
            "outer encoder input dim",
        )
        initializer(self)

    @overrides
    def forward(
        self,
        clauses: Dict[str, torch.LongTensor],
        labels: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        clauses_mask = util.get_text_field_mask(clauses, 1)
        outer_mask = util.get_text_field_mask(clauses)
        batch_size, n_clauses, n_words = clauses_mask.shape

        embedded_clauses = self.text_field_embedder(clauses)
        if self.dropout:
            embedded_clauses = self.dropout(embedded_clauses)
        embedded_clauses = embedded_clauses.view(
            batch_size * n_clauses, n_words, -1
        )
        clauses_mask = inner_mask.view(batch_size * n_clauses, n_words)
        clauses_encoded = self.inner_encoder(embedded_clauses, inner_mask)
        if self.dropout:
            clauses_encoded = self.dropout(inner_encoded)

        # get outer encodings
        clauses_encoded = inner_encoded.view(batch_size, n_clauses, -1)
        outer_encoded = self.outer_encoder(clauses_encoded, outer_mask)
        if self.dropout:
            outer_encoded = self.dropout(outer_encoded)

        # project and CRF
        logits = self.label_projection_layer(outer_encoded)
        predicted_labels = self.crf.viterbi_tags(logits, outer_mask)

        output = {
            "logits": logits,
            "clauses_mask": inner_mask,
            "outer_mask": outer_mask,
            "labels": predicted_labels,
        }

        if labels is not None:
            log_likelihood = self.crf(logits, labels, outer_mask)
            output["loss"] = -log_likelihood
            predicted_labels = [
                pad_sequence_to_length(x, n_clauses) for x in predicted_labels
            ]
            predicted_labels = torch.LongTensor(predicted_labels)
            for metric in self.metrics.values():
                metric(predicted_labels, labels, outer_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
