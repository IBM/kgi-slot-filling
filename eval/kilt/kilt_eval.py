# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in this directory of this source tree.


import argparse
import pprint
import re
import string
import ujson as json

from collections import Counter


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


# utility to get gold answers
def get_gold_answers(gold):
    if "answers" in gold:
        return set([a.strip() for a in gold["answers"] if len(a.strip()) > 0])
    ground_truths = set()
    for item in gold["output"]:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths


# utility to get max
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# EM score definition
def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _calculate_metrics(gold_records, guess_records):

    assert len(gold_records) == len(
        guess_records
    ), "different size gold: {} guess: {}".format(len(gold_records), len(guess_records))

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0

    for guess_item, gold_item in zip(guess_records, gold_records):

        # check ids
        assert (
            str(gold_item["id"]).strip() == str(guess_item["id"]).strip()
        ), "Items must have same order with same IDs"

        total_count += 1
        # check if each output of guess file exist in set of candidate answers
        gold_candidate_answers = get_gold_answers(gold_item)

        guess_answer = str(guess_item["predictions"][0]).strip()

        if len(guess_answer) == 0:
            # empty answer
            continue

        # 0. accuracy = strict exact match
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )
        normalized_em += local_em

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, guess_answer, gold_candidate_answers
        )
        normalized_f1 += local_f1

    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count

    return {
        "downstream": {
            "accuracy": accuracy,
            "em": normalized_em,
            "f1": normalized_f1,
        },
    }


def validate_input(gold_records, guess_records):

    if len(gold_records) != len(guess_records):
        print(
            "WARNING: DIFFERENT SIZE gold: {} guess: {}".format(
                len(gold_records), len(guess_records)
            )
        )

    # align order
    gold_ids = []
    for gold in gold_records:
        assert str(gold["id"]).strip() not in gold_ids, "Gold IDs should be unique"
        gold_ids.append(str(gold["id"]).strip())

    id2guess_record = {}
    for guess in guess_records:
        assert (
            str(guess["id"]).strip() not in id2guess_record
        ), "Prediction IDs should be unique"
        id2guess_record[str(guess["id"]).strip()] = guess

    guess_records = []
    for id in gold_ids:
        if id in id2guess_record:
            guess_records.append(id2guess_record[id])
        else:
            raise ValueError("ERROR: no prediction provided for id: {}".format(id))

    return gold_records, guess_records


def evaluate(gold, guess):
    pp = pprint.PrettyPrinter(indent=4)

    gold_records = load_data(gold)
    guess_records = load_data(guess)

    # 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. downstream + kilt
    result = _calculate_metrics(gold_records, guess_records)

    pp.pprint(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("guess", help="Guess KILT file")
    parser.add_argument("gold", help="Gold KILT file")

    args = parser.parse_args()
    evaluate(args.gold, args.guess)

"""
python /root/wksp/retrieve-write/rag/slot_filling/kilt_eval.py \
/data/slot_filling/structured_zeroshot-dev-kilt_predictions_token_nq.jsonl \
/data/slot_filling/structured_zeroshot-dev-kilt.jsonl 
{'downstream': {'accuracy': 0.006981740064446832, 'em': 0.1933404940923738, 'f1': 0.24537948923018685}}
                      
python /root/wksp/retrieve-write/rag/slot_filling/kilt_eval.py \
/data/slot_filling/structured_zeroshot-dev-kilt_predictions_token_nq_e1.jsonl \
/data/slot_filling/structured_zeroshot-dev-kilt.jsonl 
{'downstream': {'accuracy': 0.374328678839957, 'em': 0.3805048335123523, 'f1': 0.4785610245465251}}

python /root/wksp/retrieve-write/rag/slot_filling/rag_apply.py \
  --kilt_data /data/slot_filling/zsRE/structured_zeroshot-dev-kilt.jsonl \
  --output /data/slot_filling/predictions/structured_zeroshot-dev-kilt_predictions_token_nq_e2.jsonl \
  --model_name facebook/rag-token-nq \
  --model_path /data/slot_filling/models/zsRE_token-nq_e2
{'downstream': {'accuracy': 0.38721804511278196, 'em': 0.39151450053705694, 'f1': 0.4694302624216704}}

                     
python /root/wksp/retrieve-write/rag/slot_filling/kilt_eval.py \
/data/slot_filling/trex-dev-kilt_predictions_token_nq.jsonl \
/data/slot_filling/trex-dev-kilt.jsonl
{'downstream': {'accuracy': 0.0304, 'em': 0.15, 'f1': 0.20015754689754647}}

python /root/wksp/retrieve-write/rag/slot_filling/kilt_eval.py \
/data/slot_filling/trex-dev-kilt_predictions_token_nq_500k.jsonl \
/data/slot_filling/trex-dev-kilt.jsonl
{'downstream': {'accuracy': 0.6328, 'em': 0.634, 'f1': 0.6767234199134188}}
"""
