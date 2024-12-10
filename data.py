from Levenshtein import ratio, distance
from models import G2PModel

ANCHOR_CHAR = "▁"


def prepare_data(sent_path, lb_path=None, pos_path=None, max_samples=None):
    raw_texts = open(sent_path, encoding="utf-8").read().rstrip().split("\n")
    query_ids = [raw.index(ANCHOR_CHAR) for raw in raw_texts]
    texts = [raw.replace(ANCHOR_CHAR, "") for raw in raw_texts]
    if lb_path is None:
        if max_samples:
            return texts[:max_samples], query_ids[:max_samples], None, None

        return texts, query_ids, None, None
    else:
        phonemes = open(lb_path, encoding="utf-8").read().rstrip().split("\n")
        pos_tags = (
            open(pos_path, encoding="utf-8").read().rstrip().split("\n")
            if pos_path
            else None
        )

        if max_samples:
            texts = texts[:max_samples]
            query_ids = query_ids[:max_samples]
            phonemes = phonemes[:max_samples]
            pos_tags = pos_tags[:max_samples] if pos_tags else None

        return texts, query_ids, phonemes, pos_tags


def calculate_accuracy(
    predictions, test_texts, test_query_ids, ground_truths, pos_tags=None
):
    """Calculates accuracy and Levenshtein distance for single-phoneme predictions.

    Args:
      predictions: A list of predicted phonemes.
      ground_truths: A list of ground truth phonemes.
      pos_tags: A list of part-of-speech tags.

    Returns:
      (Accuracy score, Levenshtein distance)
    """
    total_predictions = len(predictions)
    correct_predictions = 0
    leven_distance = 0

    for text, query_id, true_phoneme, pred_phonemes in zip(
        test_texts, test_query_ids, ground_truths, predictions
    ):
        if len(pred_phonemes) > query_id:
            predicted_phoneme = pred_phonemes[query_id]
            leven_distance += 1 - ratio(true_phoneme, predicted_phoneme) if predicted_phoneme else 1
            if pred_phonemes[query_id] == true_phoneme:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    leven_distance = leven_distance / total_predictions

    return accuracy, leven_distance


def calculate_per(
    predictions, test_texts, test_query_ids, ground_truths, pos_tags=None
):
    """Calculates Phoneme Error Rate (PER) for single-phoneme predictions.

    Args:
      predictions: A list of predicted phonemes.
      ground_truths: A list of ground truth phonemes.
      pos_tags: A list of part-of-speech tags.

    Returns:
      PER score
    """
    total_phonemes = 0
    total_errors = 0

    for text, query_id, true_phoneme, pred_phonemes in zip(
        test_texts, test_query_ids, ground_truths, predictions
    ):
        if len(pred_phonemes) > query_id:
            predicted_phoneme = pred_phonemes[query_id]
            total_phonemes += len(true_phoneme)
            total_errors += distance(true_phoneme, predicted_phoneme) if predicted_phoneme else len(true_phoneme)

    return total_errors / total_phonemes if total_phonemes > 0 else 0


def test(
    model: G2PModel, sent_path="data/test.sent", lb_path="data/test.lb", pos_path=None
):
    test_texts, test_query_ids, test_phonemes, test_pos = prepare_data(
        sent_path, lb_path, pos_path
    )
    predictions = model(test_texts)

    acc, distance = calculate_accuracy(
        predictions, test_texts, test_query_ids, test_phonemes, test_pos
    )

    per = calculate_per(
        predictions, test_texts, test_query_ids, test_phonemes, test_pos
    )

    print(f"Accuracy: {acc:.2f}")
    print(f"Levenshtein Distance: {distance:.2f}")
    print(f"Phoneme Error Rate (PER): {per:.2f}")
