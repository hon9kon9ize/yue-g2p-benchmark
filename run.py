from models import (
    PyCantoneseModel,
    ToJyutpingModel,
    CantoneseG2PWModel,
    # GoogleTranslateModel,
    # FunAudioModel,
)
import time
from data import prepare_data, calculate_accuracy, calculate_per
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--max-samples", type=int, default=None)
    args = args.parse_args()

    model_classes = [
        PyCantoneseModel,
        ToJyutpingModel,
        CantoneseG2PWModel,
    ]
    model_names = []

    wordshk_texts, wordshk_query_ids, wordshk_phonemes, _ = prepare_data(
        "data/wordshk.sent", "data/wordshk.lb", args.max_samples
    )
    colloquial_texts, colloquial_query_ids, colloquial_phonemes, colloquial_pos = (
        prepare_data(
            "data/colloquial.sent", "data/colloquial.lb", "data/colloquial.pos"
        )
    )
    classical_texts, classical_query_ids, classical_phonemes, _ = prepare_data(
        "data/classical.sent", "data/classical.lb"
    )

    test_texts = wordshk_texts + colloquial_texts + classical_texts
    test_query_ids = wordshk_query_ids + colloquial_query_ids + classical_query_ids
    test_phonemes = wordshk_phonemes + colloquial_phonemes + classical_phonemes
    test_pos = colloquial_pos

    results = {}

    for model_class in model_classes:
        model = model_class()
        model_name = model.get_name()

        print(f"\nTesting {model_name}...")
        start_time = time.time()
        predictions = model(test_texts)
        runtime = time.time() - start_time

        acc, distance = calculate_accuracy(
            predictions, test_texts, test_query_ids, test_phonemes, test_pos
        )
        per = calculate_per(
            predictions, test_texts, test_query_ids, test_phonemes, test_pos
        )

        print(f"Accuracy: {acc:.4f}")
        print(f"Levenshtein Distance: {distance:.4f}")
        print(f"Phoneme Error Rate (PER): {per:.4f}")
        print(f"Runtime: {runtime:.4f}s")

        model_names.append(model_name)
        results[model_name] = {
            "accuracy": acc,
            "distance": distance,
            "per": per,
            "runtime": runtime,
        }

    colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
    ]  # Define a list of color-blind friendly colors
    width = 0.35

    # Plot the results
    plt.figure(figsize=(14, 8))
    x = range(len(model_names))
    accuracy_bars = plt.bar(
        x,
        [results[model_name]["accuracy"] for model_name in model_names],
        width,
        label="Accuracy",
        color=colors[0],
    )
    distance_bars = plt.bar(
        [i + width for i in x],
        [results[model_name]["per"] for model_name in model_names],
        width,
        label="Phoneme Error Rate (PER)",
        color=colors[1],
    )
    plt.xticks([i + width / 2 for i in x], model_names)

    plt.ylabel("Scores")
    plt.title("G2P Model Performance")
    plt.legend()

    # Add text labels on the bars
    for bar in accuracy_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )
    for bar in distance_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )

    # save to result.png
    plt.savefig("result.png")

    # Plot the runtime
    plt.figure(figsize=(14, 8))
    runtime_bars = plt.bar(
        model_names,
        [results[model_name]["runtime"] for model_name in model_names],
        width,
        label="Runtime",
        color=colors[2],
    )

    plt.ylabel("Runtime (s)")
    plt.title("G2P Model Runtime")
    plt.legend()

    # Add text labels on the bars
    for bar in runtime_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )

    # save to runtime.png
    plt.savefig("runtime.png")
