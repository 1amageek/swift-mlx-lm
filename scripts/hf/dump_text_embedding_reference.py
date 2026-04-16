#!/usr/bin/env python3
"""Generate text-embedding reference vectors with mlx_embeddings.

Usage:
    python3 scripts/hf/dump_text_embedding_reference.py \
        --model-dir ~/.cache/huggingface/hub/.../snapshot \
        --dataset Tests/SwiftLMTests/TestData/text_embedding_smoke_dataset.json \
        --output /tmp/embedding_reference.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Local HuggingFace snapshot directory")
    parser.add_argument("--dataset", required=True, help="Dataset JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompts(model_dir: Path) -> dict[str, str]:
    config_path = model_dir / "config_sentence_transformers.json"
    if not config_path.exists():
        return {}
    config = load_json(config_path)
    prompts = config.get("prompts", {})
    if not isinstance(prompts, dict):
        raise ValueError("config_sentence_transformers.json prompts must be an object")
    return {str(key): str(value) for key, value in prompts.items()}


def render_texts(
    items: Iterable[dict],
    prompt_name: str | None,
    prompts: dict[str, str],
) -> list[str]:
    prefix = ""
    if prompt_name is not None:
        if prompt_name not in prompts:
            raise ValueError(f"Prompt '{prompt_name}' was requested by the dataset but is missing from the model config")
        prefix = prompts[prompt_name]
    rendered = []
    for item in items:
        rendered.append(prefix + str(item["text"]))
    return rendered


def batched_embeddings(model, tokenizer, texts: list[str], batch_size: int) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="mlx")
        outputs = model(encoded["input_ids"], attention_mask=encoded["attention_mask"])
        batch_embeddings = outputs.text_embeds.tolist()
        for embedding in batch_embeddings:
            embeddings.append([float(value) for value in embedding])
    return embeddings


def l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    lhs_norm = l2_norm(lhs)
    rhs_norm = l2_norm(rhs)
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    dot = sum(left * right for left, right in zip(lhs, rhs))
    return dot / (lhs_norm * rhs_norm)


def evaluate_metrics(
    dataset: dict,
    query_embeddings: dict[str, list[float]],
    document_embeddings: dict[str, list[float]],
) -> dict[str, float]:
    query_results = []
    for query in dataset["queries"]:
        relevant_ids = set(query["relevantDocumentIDs"])
        ranked_documents = sorted(
            (
                {
                    "id": document["id"],
                    "score": cosine_similarity(
                        query_embeddings[query["id"]],
                        document_embeddings[document["id"]],
                    ),
                }
                for document in dataset["documents"]
            ),
            key=lambda item: (-item["score"], item["id"]),
        )
        first_relevant_index = next(
            index for index, item in enumerate(ranked_documents) if item["id"] in relevant_ids
        )
        relevant_scores = [item["score"] for item in ranked_documents if item["id"] in relevant_ids]
        irrelevant_scores = [item["score"] for item in ranked_documents if item["id"] not in relevant_ids]
        top3 = ranked_documents[:3]
        relevant_in_top3 = sum(1 for item in top3 if item["id"] in relevant_ids)
        query_results.append(
            {
                "top1Hit": ranked_documents[0]["id"] in relevant_ids,
                "reciprocalRank": 1.0 / float(first_relevant_index + 1),
                "recallAt3": float(relevant_in_top3) / float(len(relevant_ids)),
                "relevantMargin": max(relevant_scores) - max(irrelevant_scores),
            }
        )

    query_count = float(max(len(query_results), 1))
    return {
        "top1Accuracy": sum(1.0 for result in query_results if result["top1Hit"]) / query_count,
        "meanReciprocalRank": sum(result["reciprocalRank"] for result in query_results) / query_count,
        "meanRecallAt3": sum(result["recallAt3"] for result in query_results) / query_count,
        "meanRelevantMargin": sum(result["relevantMargin"] for result in query_results) / query_count,
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    dataset = load_json(dataset_path)
    prompts = load_prompts(model_dir)
    query_prompt_name = dataset.get("queryPromptName")
    document_prompt_name = dataset.get("documentPromptName")

    try:
        from mlx_embeddings import load
    except ImportError as error:
        raise SystemExit(
            "mlx_embeddings is required. Install it with `pip install mlx-embeddings`."
        ) from error

    model, tokenizer = load(str(model_dir))

    document_texts = render_texts(dataset["documents"], document_prompt_name, prompts)
    query_texts = render_texts(dataset["queries"], query_prompt_name, prompts)
    document_vectors = batched_embeddings(model, tokenizer, document_texts, args.batch_size)
    query_vectors = batched_embeddings(model, tokenizer, query_texts, args.batch_size)

    document_embeddings = {
        document["id"]: embedding
        for document, embedding in zip(dataset["documents"], document_vectors, strict=True)
    }
    query_embeddings = {
        query["id"]: embedding
        for query, embedding in zip(dataset["queries"], query_vectors, strict=True)
    }

    if document_vectors:
        embedding_dimension = len(document_vectors[0])
    elif query_vectors:
        embedding_dimension = len(query_vectors[0])
    else:
        raise SystemExit("The dataset did not contain any documents or queries.")

    metrics = evaluate_metrics(dataset, query_embeddings, document_embeddings)
    payload = {
        "dataset": dataset,
        "embeddingDimension": embedding_dimension,
        "documentEmbeddings": document_embeddings,
        "queryEmbeddings": query_embeddings,
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(
        "[EmbeddingReference] "
        f"dataset={dataset['name']} docs={len(dataset['documents'])} queries={len(dataset['queries'])} "
        f"top1={metrics['top1Accuracy']:.3f} mrr={metrics['meanReciprocalRank']:.3f} "
        f"recall@3={metrics['meanRecallAt3']:.3f} margin={metrics['meanRelevantMargin']:.3f} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
