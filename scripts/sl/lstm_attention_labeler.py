"""
Originally from:
https://github.com/handsomezebra/nlp/blob/master/my_library/models/sequence_classification.py
https://github.com/handsomezebra/nlp/blob/master/my_library/models/self_attentive_lstm.py

Works like a charm OOTB.
"""

from typing import Dict, Optional, List

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
)
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import ConditionalRandomField
from allennlp.nn import util
from allennlp.training.metrics import (
    CategoricalAccuracy,
    F1Measure,
    SpanBasedF1Measure,
)
from allennlp.modules import InputVariationalDropout, FeedForward
from allennlp.nn.util import (
    masked_softmax,
    weighted_sum,
    sequence_cross_entropy_with_logits,
    get_text_field_mask,
)
from torch.nn import Dropout, Linear


@Model.register("sl_model")
class SequenceLabeler(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        embedding_dropout: float,
        seq2seq_encoder: Seq2SeqEncoder,
        initializer: InitializerApplicator = InitializerApplicator(),
        loss_weights: Optional[List] = [],
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(SequenceLabeler, self).__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.seq2seq_encoder = seq2seq_encoder
        self.self_attentive_pooling_projection = nn.Linear(
            seq2seq_encoder.get_output_dim(), 1
        )
        self._classifier = nn.Linear(
            in_features=seq2seq_encoder.get_output_dim(),
            out_features=vocab.get_vocab_size("labels"),
        )
        self._crf = ConditionalRandomField(vocab.get_vocab_size("labels"))
        self.loss = torch.nn.CrossEntropyLoss()
        self._f1 = SpanBasedF1Measure(vocab, "labels")
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        initializer(self)

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        mask_tokens = util.get_text_field_mask(tokens)
        embedded_tokens = self.text_field_embedder(tokens)
        dropped_embedded_tokens = self._embedding_dropout(embedded_tokens)

        encoded_tokens = self.seq2seq_encoder(dropped_embedded_tokens, mask_tokens)
        self_attentive_logits = self.self_attentive_pooling_projection(
            encoded_tokens
        ).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, mask_tokens)
        encoding_result = util.weighted_sum(encoded_tokens, self_weights)
        classified = self._classifier(encoding_result)
        viterbi_tags = self._crf.viterbi_tags(classified, mask)
        viterbi_tags = [path for path, score in viterbi_tags]
        broadcasted = self._broadcast_tags(viterbi_tags, classified)

        log_likelihood = self._crf(classified, label, mask_tokens)
        self._f1(broadcasted, label, mask_tokens)

        output: Dict[str, torch.Tensor] = {}
        output["loss"] = -log_likelihood
        output["logits"] = broadcasted

        if label is not None:
            # label = torch.max(label, 1)[1]
            output["loss"] = sequence_cross_entropy_with_logits(
                classified, label, mask_tokens
            )
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._f1.get_metric(reset)
        # return self._accuracy.get_metric(reset)


@Seq2SeqEncoder.register("masked_multi_head_self_attention")
class MaskedMultiHeadSelfAttention(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .
    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.
    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        attention_dim: int,
        values_dim: int,
        output_projection_dim: int = None,
        attention_dropout_prob: float = 0.1,
    ) -> None:
        super(MaskedMultiHeadSelfAttention, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"Key size ({attention_dim}) must be divisible by the number of "
                f"attention heads ({num_heads})."
            )

        if values_dim % num_heads != 0:
            raise ValueError(
                f"Value size ({values_dim}) must be divisible by the number of "
                f"attention heads ({num_heads})."
            )

        self._combined_projection = Linear(input_dim, 2 * attention_dim + values_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(
        self,  # pylint: disable=arguments-differ
        inputs: torch.Tensor,
        mask: torch.LongTensor = None,
    ) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).
        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)

        # Shape (batch_size, timesteps, 2 * attention_dim + values_dim)
        combined_projection = self._combined_projection(inputs)
        combined_projection *= mask.unsqueeze(-1).float()

        # split by attention dim - if values_dim > attention_dim, we will get more
        # than 3 elements returned. All of the rest are the values vector, so we
        # just concatenate them back together again below.
        queries, keys, *values = combined_projection.split(self._attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        # Shape (num_heads * batch_size, timesteps, values_dim / num_heads)
        values_per_head = values.view(
            batch_size, timesteps, num_heads, int(self._values_dim / num_heads)
        )
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(
            batch_size * num_heads, timesteps, int(self._values_dim / num_heads)
        )

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(
            batch_size, timesteps, num_heads, int(self._attention_dim / num_heads),
        )
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(
            batch_size * num_heads, timesteps, int(self._attention_dim / num_heads),
        )

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        keys_per_head = keys.view(
            batch_size, timesteps, num_heads, int(self._attention_dim / num_heads),
        )
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(
            batch_size * num_heads, timesteps, int(self._attention_dim / num_heads),
        )

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = (
            torch.bmm(queries_per_head, keys_per_head.transpose(1, 2)) / self._scale
        )

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        expanded_mask = mask.repeat(1, num_heads).view(
            batch_size * num_heads, timesteps
        )
        expanded_mask = expanded_mask.unsqueeze(-1) * expanded_mask.unsqueeze(-2)
        attention = masked_softmax(scaled_similarities, expanded_mask)
        attention = self._attention_dropout(attention)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, timesteps, values_dim/num_heads)
        outputs = weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, timesteps, values_dim)
        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(
            batch_size, num_heads, timesteps, int(self._values_dim / num_heads)
        )
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        outputs *= mask.unsqueeze(-1).float()
        return outputs


@Seq2SeqEncoder.register("self_attentive_lstm")
class SelfAttentiveLstm(Seq2SeqEncoder):
    def __init__(
        self,
        lstm_encoder: Seq2SeqEncoder,
        self_attention_encoder: Seq2SeqEncoder,
        projection_feedforward: FeedForward,
        dropout: float = 0.5,
    ) -> None:
        super(SelfAttentiveLstm, self).__init__()

        self._lstm_encoder = lstm_encoder
        self._self_attention_encoder = self_attention_encoder
        self._projection_feedforward = projection_feedforward

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

    def get_input_dim(self):
        return self._lstm_encoder.get_input_dim()

    def get_output_dim(self):
        return self._projection_feedforward.get_output_dim()

    @overrides
    def forward(
        self,  # pylint: disable=arguments-differ
        inputs: torch.Tensor,
        mask: torch.LongTensor = None,
    ) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).
        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_dim),
        """
        # apply dropout for LSTM

        inputs = inputs * mask.unsqueeze(-1).float()

        if self.rnn_input_dropout:
            inputs = self.rnn_input_dropout(inputs)

        # encode inputs in two different ways
        lstm_encoded_inputs = self._lstm_encoder(inputs, mask)

        # Shape: (batch_size, timesteps, timesteps)
        self_attention_encoded_inputs = self._self_attention_encoder(
            lstm_encoded_inputs, mask
        )

        # the "enhancement" layer
        enhanced = torch.cat(
            [inputs, lstm_encoded_inputs, self_attention_encoded_inputs], dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected = self._projection_feedforward(enhanced)

        return projected
