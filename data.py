from Levenshtein import ratio
from models import G2PModel

ANCHOR_CHAR = "▁"


def prepare_data(sent_path, lb_path=None, max_samples=None):
    raw_texts = open(sent_path, encoding="utf-8").read().rstrip().split("\n")
    query_ids = [raw.index(ANCHOR_CHAR) for raw in raw_texts]
    texts = [raw.replace(ANCHOR_CHAR, "") for raw in raw_texts]
    if lb_path is None:
        if max_samples:
            return texts[:max_samples], query_ids[:max_samples], None

        return texts, query_ids, None
    else:
        phonemes = open(lb_path, encoding="utf-8").read().rstrip().split("\n")

        if max_samples:
            texts = texts[:max_samples]
            query_ids = query_ids[:max_samples]
            phonemes = phonemes[:max_samples]

        return texts, query_ids, phonemes


def calculate_accuracy(predictions, test_texts, test_query_ids, ground_truths):
    """Calculates accuracy, recall, and F1 score for single-phoneme predictions.

    Args:
      predictions: A list of predicted phonemes.
      ground_truths: A list of ground truth phonemes.

    Returns:
      (Accuracy score, Levenshtein distance)
    """
    total_predictions = len(predictions)
    correct_predictions = 0
    leven_distance = 0

    for i, (text, query_id, phoneme, pred) in enumerate(
        zip(test_texts, test_query_ids, ground_truths, predictions)
    ):
        pred_phonemes = pred.split(" ")

        if len(pred_phonemes) > query_id:
            leven_distance += 1 - ratio(phoneme, pred_phonemes[query_id])
            if pred_phonemes[query_id] == phoneme:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    leven_distance = leven_distance / total_predictions

    return accuracy, leven_distance


def test(model: G2PModel, sent_path="data/test.sent", lb_path="data/test.lb"):
    test_texts, test_query_ids, test_phonemes = prepare_data(sent_path, lb_path)
    predictions = model(test_texts)

    acc, distance = calculate_accuracy(
        predictions, test_texts, test_query_ids, test_phonemes
    )

    print(f"Accuracy: {acc:.2f}")
    print(f"Levenshtein Distance: {distance:.2f}")
