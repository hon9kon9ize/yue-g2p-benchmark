from Levenshtein import ratio, distance
from sklearn.metrics import precision_score, recall_score, f1_score
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

    for text, query_id, true_phoneme, pred in zip(
        test_texts, test_query_ids, ground_truths, predictions
    ):
        pred_phonemes = pred.split(" ")
        if len(pred_phonemes) > query_id:
            predicted_phoneme = pred_phonemes[query_id]
            total_phonemes += len(true_phoneme)
            total_errors += distance(
                true_phoneme, predicted_phoneme
            )  # Raw edit distance

    return total_errors / total_phonemes if total_phonemes > 0 else 0


def calculate_wer(predictions, test_texts, test_query_ids, ground_truths):
    """Calculates Word Error Rate (WER)."""
    total_words = len(predictions)
    errors = 0

    for pred, true_phoneme, query_id in zip(predictions, ground_truths, test_query_ids):
        pred_phonemes = pred.split(" ")
        if len(pred_phonemes) > query_id and pred_phonemes[query_id] != true_phoneme:
            errors += 1
        elif len(pred_phonemes) <= query_id:
            errors += 1

    return errors / total_words

def calculate_pos_accuracy(
    predictions, test_texts, test_query_ids, ground_truths, pos_tags
):
    """Calculates accuracy per POS tag."""
    pos_dict = {}

    for pred, true_phoneme, pos, query_id in zip(
        predictions, ground_truths, pos_tags, test_query_ids
    ):
        pred_phonemes = pred.split(" ")
        correct = False
        if len(pred_phonemes) > query_id and pred_phonemes[query_id] == true_phoneme:
            correct = True

        if pos not in pos_dict:
            pos_dict[pos] = {"correct": 0, "total": 0}

        pos_dict[pos]["total"] += 1
        if correct:
            pos_dict[pos]["correct"] += 1

    pos_accuracy = {
        pos: data["correct"] / data["total"] for pos, data in pos_dict.items()
    }
    return pos_accuracy


def calculate_cer(predictions, test_texts, test_query_ids, ground_truths):
    """Calculates Character Error Rate (CER)."""
    total_chars = 0
    total_errors = 0

    for pred, true_phoneme, query_id in zip(predictions, ground_truths, test_query_ids):
        pred_phonemes = pred.split(" ")
        if len(pred_phonemes) > query_id:
            predicted_phoneme = pred_phonemes[query_id]
        else:
            predicted_phoneme = ""
        total_chars += len(true_phoneme)
        total_errors += distance(true_phoneme, predicted_phoneme)

    return total_errors / total_chars if total_chars > 0 else 0


def calculate_precision_recall_f1(
    predictions, test_texts, test_query_ids, ground_truths
):
    """Calculates Phoneme Precision, Recall, and F1-Score."""
    y_true = []
    y_pred = []

    for pred, true_phoneme, query_id in zip(predictions, ground_truths, test_query_ids):
        pred_phonemes = pred.split(" ")
        if len(pred_phonemes) > query_id:
            predicted_phoneme = pred_phonemes[query_id]
        else:
            predicted_phoneme = ""

        # Ensure both true and predicted phonemes are of the same length
        min_length = min(len(true_phoneme), len(predicted_phoneme))
        y_true.extend(list(true_phoneme[:min_length]))
        y_pred.extend(list(predicted_phoneme[:min_length]))

        # If true_phoneme is longer, pad y_pred with empty strings
        if len(true_phoneme) > min_length:
            y_true.extend(list(true_phoneme[min_length:]))
            y_pred.extend([""] * (len(true_phoneme) - min_length))

        # If predicted_phoneme is longer, pad y_true with empty strings
        if len(predicted_phoneme) > min_length:
            y_true.extend([""] * (len(predicted_phoneme) - min_length))
            y_pred.extend(list(predicted_phoneme[min_length:]))

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return precision, recall, f1


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
    wer = calculate_wer(predictions, test_texts, test_query_ids, test_phonemes)

    pos_accuracy = calculate_pos_accuracy(
        predictions, test_texts, test_query_ids, test_phonemes, test_pos
    )
    cer = calculate_cer(predictions, test_texts, test_query_ids, test_phonemes)

    precision, recall, f1 = calculate_precision_recall_f1(
        predictions, test_texts, test_query_ids, test_phonemes
    )

    print(f"Accuracy: {acc:.2f}")
    print(f"Levenshtein Distance: {distance:.2f}")
    print(f"Phoneme Error Rate (PER): {per:.2f}")
    print(f"Word Error Rate (WER): {wer:.2f}")
    print(f"Sentence Error Rate (SER): {ser:.2f}")
    print(f"Character Error Rate (CER): {cer:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
