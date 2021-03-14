from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("jcc_predictor")
class JCCPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        id_ = json_dict["id"]
        clauses = json_dict["clauses"]
        labels = json_dict["labels"]
        instance = self._dataset_reader.text_to_instance(
            id_=id_, clauses=clauses, labels=labels
        )
        label_dict = self._model.vocab.get_index_to_token_vocabulary("labels")
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        return instance, {"all_labels": all_labels}
