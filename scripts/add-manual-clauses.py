import fileinput
import json
import sys


WEIRD_CHAR = "⌇"


def fix_anno_tokenization(clauses, tokens):
    # -> [["They", "'re", "bad"], ["she", "said"]]
    # [["They're", "bad"], ["she", "said"]], ["They", "'re", "bad", "she", "said"]
    result = []
    tokit = iter(tokens)
    for clause in clauses:
        clause_result = []
        for cloken in clause:
            token = next(tokit)
            if token == cloken:
                clause_result.append(cloken)
                continue

            assert len(token) < len(cloken), "<{}> vs <{}>".format(
                token, cloken
            )
            while cloken and len(cloken) > len(token):
                clause_result.append(token)
                cloken = cloken[len(token) :]
                token = next(tokit)
            else:
                clause_result.append(token)
        result.append(clause_result)
    return result


def process_annotations(anno):
    clauses = [
        [
            word.strip()
            .replace("&lt", "<")
            .replace("&amp", "&")
            .replace("&gt", ">")
            for word in clause.strip().split(" ")
            if word.strip()
        ]
        for clause in anno.strip().split(WEIRD_CHAR)
    ]
    return [clause for clause in clauses if clause]


def align_tags(clauses, tokens):
    tokens, tags = list(zip(*tokens))
    clauses = fix_anno_tokenization(clauses, tokens)
    rtags = list(reversed(tags))
    clauselen = sum(len(clause) for clause in clauses)
    assert clauselen == len(
        tags
    ), f"WTF: {clauses}, {tags}, {len(tags)}, {clauselen}"

    return [[(token, rtags.pop()) for token in clause] for clause in clauses]


id2annotation = {}
with open("annotations/clauses-annotated.tsv") as f:

    for line in f:
        id_, orig, anno = line.strip().split("\t")
        id2annotation[id_] = process_annotations(anno)

for line in fileinput.input():
    data = json.loads(line)
    annos = id2annotation.pop(data["id"], None)
    if annos:
        try:
            data["clauses-manual"] = align_tags(annos, data["tokens"])
        except AssertionError as e:
            print(data["text"], file=sys.stderr)
            print(data["tokens"], file=sys.stderr)
            raise e
    print(json.dumps(data))
assert not annos
