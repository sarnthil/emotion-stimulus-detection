from collections import defaultdict
from itertools import zip_longest
from sklearn.metrics import cohen_kappa_score


datasets = ["emotion-stimulus", "eca", "gne", "electoral_tweets"]  # , "reman"


def annotator_dict_from_file(filename):
    annotator = defaultdict(dict)
    with open(filename) as f:
        for line in f:
            # id text clauses
            annotations = line.split("\t")
            id_ = annotations[0]
            text = annotations[1]
            clauses = annotations[2].split("⌇")
            ds, _, __ = id_.partition("-")
            annotator[ds][id_] = clauses
    return annotator


annotators = [
    annotator_dict_from_file("annotations/clauses_a1.tsv"),
    annotator_dict_from_file("annotations/clauses_a2.tsv"),
]


def token_clause_boundaries(clauses):
    for clause in clauses:
        for i, _ in enumerate(clause.split(" ")):
            yield i == 0


kappas = {}
for dataset in annotators[0]:
    vector_1, vector_2 = [], []
    for id_, clauses_1 in annotators[0][dataset].items():
        clauses_2 = annotators[1][dataset][id_]
        for x, y in zip(token_clause_boundaries(clauses_1), token_clause_boundaries(clauses_2)):
            vector_1.append(x)
            vector_2.append(y)
    kappas[dataset] = cohen_kappa_score(vector_1, vector_2)

print(kappas)
