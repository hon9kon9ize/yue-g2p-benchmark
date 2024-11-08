from models import (
    PyCantoneseModel,
    ToJyutpingModel,
    CantoneseG2PWModel,
    GoogleTranslateModel,
)
import time
from data import prepare_data, calculate_accuracy
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sent-path", type=str, default="data/test.sent")
    args.add_argument("--lb-path", type=str, default="data/test.lb")
    args.add_argument("--max-samples", type=int, default=None)
    args = args.parse_args()

    model_classes = [
        PyCantoneseModel,
        ToJyutpingModel,
        GoogleTranslateModel,
        CantoneseG2PWModel,
    ]
    model_names = []

    test_texts, test_query_ids, test_phonemes = prepare_data(
        args.sent_path, args.lb_path, args.max_samples
    )
    results = {}

    for model_class in model_classes:
        model = model_class()
        model_name = model.get_name()

        print(f"\nTesting {model_name}...")
        start_time = time.time()
        predictions = model(test_texts)
        runtime = time.time() - start_time

        acc, distance = calculate_accuracy(
            predictions, test_texts, test_query_ids, test_phonemes
        )

        print(f"Accuracy: {acc:.4f}")
        print(f"Levenshtein Distance: {distance:.4f}")

        model_names.append(model_name)
        results[model_name] = {
            "accuracy": acc,
            "distance": distance,
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
    plt.bar(
        x,
        [results[model_name]["accuracy"] for model_name in model_names],
        width,
        label="Accuracy",
        color=colors[0],
    )
    plt.bar(
        [i + width for i in x],
        [results[model_name]["distance"] for model_name in model_names],
        width,
        label="Levenshtein Distance",
        color=colors[1],
    )
    plt.xticks([i + width / 2 for i in x], model_names)

    plt.ylabel("Scores")
    plt.title("G2P Model Performance")
    plt.legend()
    # save to result.png
    plt.savefig("result.png")

    # Plot the runtime
    plt.figure(figsize=(14, 8))
    plt.bar(
        model_names,
        [results[model_name]["runtime"] for model_name in model_names],
        width,
        label="Runtime",
        color=colors[2],
    )

    plt.ylabel("Runtime (s)")
    plt.title("G2P Model Runtime")
    plt.legend()
    # save to runtime.png
    plt.savefig("runtime.png")
