import os
import sys
import json
import collections
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import spacy


nlp = spacy.load('en_core_web_md')


class ClassificationEvaluator:
    def __init__(self, labels=None):
        self.labels = labels

    def evaluate(self, y_true, y_pred):
        if not self.labels:
            self.labels = list(set(y_true))

        # micro_accuracy = sum([y_t == y_p for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)
        micro_accuracy = accuracy_score(y_true, y_pred)
        results = {}
        results["micro_accuracy"] = float("{0:.4f}".format(micro_accuracy)) #int(100 * micro_accuracy) / 100

        conf_mat = confusion_matrix(y_true, y_pred, labels=self.labels)
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        macro_accuracy = np.mean([conf_mat_norm[i][i] for i in range(conf_mat_norm.shape[0])])
        results["macro_accuracy"] = float("{0:.4f}".format(macro_accuracy)) #int(100 * macro_accuracy) / 100
        return results


class MoreEvaluator:
    def __init__(self, max_bleu_order=4, bleu_smoothing=True):
        self.max_bleu_order = max_bleu_order
        self.bleu_smoothing = bleu_smoothing

    def evaluate(self, y_true, y_pred):
        results = {}
        bleu_scores = [compute_bleu([[y.split()] for y in y_true], [y.split() for y in y_pred],
                                    max_order=bleu_order, smooth=self.bleu_smoothing)[0]
                       for bleu_order in range(1, self.max_bleu_order + 1)]

        for bleu_order, bleu_score in enumerate(bleu_scores):
            results["bleu_" + str(bleu_order + 1)] = float("{0:.4f}".format(bleu_score))
        return results


class CombinedEvaluator:
    def __init__(self, labels=['yes', 'no', 'more', 'irrelevant'], accuracy_targets=['yes', 'no', 'irrelevant']):
        self.labels = labels
        self.accuracy_targets = accuracy_targets
        self.classification_evaluator = ClassificationEvaluator(labels=labels)
        self.more_evaluator = MoreEvaluator()

    def replace_follow_up_with_more(self, y_list):
        return [y.lower() if y.lower() in self.accuracy_targets else 'more' for y in y_list]

    def extract_follow_ups(self, y_true, y_pred):
        extracted = [(y_t, y_p) for (y_t, y_p) in zip(y_true, y_pred) if
                     y_t.lower() not in self.labels and y_p.lower() not in self.labels]
        if extracted:
            return zip(*extracted)
        else:
            return [], []

    def evaluate(self, y_true, y_pred):

        # Classification
        classification_y_true = self.replace_follow_up_with_more(y_true)
        classification_y_pred = self.replace_follow_up_with_more(y_pred)
        results = self.classification_evaluator.evaluate(classification_y_true, classification_y_pred)

        # Follow Up Generation
        num_true_follow_ups = len([y_t for y_t in y_true if y_t.lower() not in self.labels])
        num_pred_follow_ups = len([y_p for y_p in y_pred if y_p.lower() not in self.labels])
        # print(f'{num_true_follow_ups} follow-ups in ground truth. {num_pred_follow_ups} follow-ups predicted | {len(generation_y_true)} follow-up questions used for BLEU evaluation.')
        generation_y_true, generation_y_pred = self.extract_follow_ups(y_true, y_pred)
        if generation_y_true and generation_y_pred:
            results.update(self.more_evaluator.evaluate(generation_y_true, generation_y_pred))
        else:
            results.update({'bleu_{}'.format(i): 0. for i in range(1, 5)})
        return results


def prepro(text):
    doc = nlp(text, disable=['parser', 'tagger', 'ner'])
    result = ""
    for token in doc:
        orth = token.text
        if orth == "":
            result += " "
        elif orth == " ":
            result += " "
        else:
            result += orth.lower() + " "
    return result.strip().replace('\n', '')


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def evaluate(gold_file, prediction_file, mode='follow_ups'):
    assert mode in ['', 'combined', 'follow_ups', 'classification'], "Mode not recognised"

    with open(gold_file, 'r') as f:
        ground_truths = json.load(f)

    with open(prediction_file, 'r') as f:
        predictions = json.load(f)

    # Check if all IDs are aligned
    # assert len(ground_truths) == len(predictions), "Predictions and ground truths have different sample sizes"

    ground_truth_map = {g["utterance_id"]: g for g in ground_truths}
    predictions_map = {p["utterance_id"]: p for p in predictions}
    for k in ground_truth_map:
        if k not in predictions_map:
            predictions_map[k] = {'utterance_id': k, 'answer': 'missing'}

    for gid in ground_truth_map:
        assert gid in predictions_map

    # Extract answers and prepro

    ground_truths = []
    predictions = []

    for uid in ground_truth_map.keys():
        ground_truths.append(prepro(ground_truth_map[uid]['answer']))
        predictions.append(prepro(predictions_map[uid]['answer']))

    if mode == 'follow_ups':
        evaluator = MoreEvaluator()
        results = evaluator.evaluate(ground_truths, predictions)

    elif mode == 'classification':
        evaluator = ClassificationEvaluator(labels=['yes', 'no', 'more', 'irrelevant'])
        results = evaluator.evaluate(ground_truths, predictions)

    else:
        evaluator = CombinedEvaluator(labels=['yes', 'no', 'more', 'irrelevant'])
        results = evaluator.evaluate(ground_truths, predictions)

    return results


if __name__ == '__main__':
    mode = 'combined'

    prediction_file = sys.argv[1]
    gold_file = sys.argv[2]

    results = evaluate(gold_file, prediction_file, mode=mode)
    print(results)
