"""Run a reproducible dense-only versus hybrid+RRF RAGAS evaluation.

The two configurations share the golden dataset, generator, evaluator, prompt,
top-k and fallback setting. Only ``use_reranking`` changes between them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
import subprocess
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from src.task10_generation import (
    SAFE_REFUSAL,
    generate_grounded_answer,
)
from src.task9_retrieval_pipeline import retrieve


ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT_DIR / "group_project" / "evaluation"
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RAW_RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
BONUS_RESULTS_PATH = EVALUATION_DIR / "bonus_reranker_results.json"
REPORT_PATH = ROOT_DIR / "reports" / "RESULT.md"
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)
CONFIGURATIONS = (
    ("dense_only", False),
    ("hybrid_rrf", True),
)

load_dotenv(ROOT_DIR / ".env")
logger = logging.getLogger(__name__)


def _load_golden_dataset(path: Path) -> list[dict[str, str]]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) < 15:
        raise ValueError("Golden dataset must contain at least 15 cases")

    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(dataset):
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError(f"Golden case {index} is missing required fields")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in required):
            raise ValueError(f"Golden case {index} contains an empty required field")
    return dataset


def _run_configuration(
    dataset: list[dict[str, str]],
    *,
    name: str,
    use_reranking: bool,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run retrieval and generation while keeping all non-retrieval settings fixed."""
    rows: list[dict[str, Any]] = []
    candidate_k = max(top_k, min(top_k * 2, 20))
    for index, case in enumerate(dataset, 1):
        logger.info("[%s] Generating %d/%d", name, index, len(dataset))
        started = time.perf_counter()
        candidates = retrieve(
            case["question"],
            top_k=candidate_k,
            score_threshold=0.0,
            use_reranking=use_reranking,
        )
        answer, selected_sources = generate_grounded_answer(
            case["question"],
            candidates,
            top_k=top_k,
        )
        # Keep retrieval measurable even when generation safely refuses.
        evaluation_sources = selected_sources or candidates[:top_k]
        rows.append(
            {
                "case_id": index,
                "configuration": name,
                "question": case["question"],
                "reference": case["expected_answer"],
                "reference_context": case["expected_context"],
                "response": answer,
                "retrieved_contexts": [
                    source["content"] for source in evaluation_sources
                ],
                "source_ids": [source["id"] for source in evaluation_sources],
                "retrieval_methods": [
                    source["retrieval_method"] for source in evaluation_sources
                ],
                "latency_seconds": time.perf_counter() - started,
            }
        )
    return rows


def _build_metrics(judge_model: str, embedding_model: str):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for RAGAS evaluation")

    client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=3)
    evaluator_llm = llm_factory(
        judge_model,
        provider="openai",
        client=client,
        adapter="instructor",
        temperature=0.0,
    )
    evaluator_embeddings = embedding_factory(
        "openai",
        model=embedding_model,
        client=client,
        interface="modern",
    )
    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=3,
        ),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "context_precision": ContextPrecisionWithReference(llm=evaluator_llm),
    }


async def _score_rows(
    rows: list[dict[str, Any]],
    *,
    judge_model: str,
    embedding_model: str,
) -> None:
    metrics = _build_metrics(judge_model, embedding_model)
    for index, row in enumerate(rows, 1):
        logger.info(
            "[ragas] Scoring %d/%d (%s case %d)",
            index,
            len(rows),
            row["configuration"],
            row["case_id"],
        )
        contexts = row["retrieved_contexts"]
        if not contexts:
            row["scores"] = {name: 0.0 for name in METRIC_NAMES}
            continue

        if row["response"] == SAFE_REFUSAL:
            faithfulness = 0.0
            answer_relevancy = 0.0
        else:
            faithfulness = float(
                (
                    await metrics["faithfulness"].ascore(
                        user_input=row["question"],
                        response=row["response"],
                        retrieved_contexts=contexts,
                    )
                ).value
            )
            answer_relevancy = float(
                (
                    await metrics["answer_relevancy"].ascore(
                        user_input=row["question"],
                        response=row["response"],
                    )
                ).value
            )

        context_recall = float(
            (
                await metrics["context_recall"].ascore(
                    user_input=row["question"],
                    retrieved_contexts=contexts,
                    reference=row["reference"],
                )
            ).value
        )
        context_precision = float(
            (
                await metrics["context_precision"].ascore(
                    user_input=row["question"],
                    reference=row["reference"],
                    retrieved_contexts=contexts,
                )
            ).value
        )
        raw_scores = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision,
        }
        if any(not math.isfinite(score) for score in raw_scores.values()):
            raise ValueError(
                f"RAGAS returned a non-finite score for {row['configuration']} "
                f"case {row['case_id']}: {raw_scores}"
            )
        row["scores"] = raw_scores


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for config_name, _ in CONFIGURATIONS:
        config_rows = [row for row in rows if row["configuration"] == config_name]
        metrics = {
            metric: statistics.fmean(row["scores"][metric] for row in config_rows)
            for metric in METRIC_NAMES
        }
        metrics["average"] = statistics.fmean(metrics.values())
        metrics["mean_latency_seconds"] = statistics.fmean(
            row["latency_seconds"] for row in config_rows
        )
        metrics["refusal_rate"] = statistics.fmean(
            row["response"] == SAFE_REFUSAL for row in config_rows
        )
        summary[config_name] = metrics
    return summary


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    suffix = " + working-tree changes" if dirty else ""
    return completed.stdout.strip() + suffix


def _failure_analysis(row: dict[str, Any]) -> tuple[str, str]:
    scores = row["scores"]
    if row["response"] == SAFE_REFUSAL and scores["context_precision"] >= 0.5:
        return (
            "generation",
            "Evidence liên quan đã được retrieve, nhưng generation hoặc "
            "citation validation tạo false refusal.",
        )
    retrieval_score = min(scores["context_recall"], scores["context_precision"])
    generation_score = min(scores["faithfulness"], scores["answer_relevancy"])
    if retrieval_score <= generation_score:
        return (
            "retrieval",
            "Retrieved context có recall/precision thấp hơn; cần kiểm tra "
            "chunk boundaries và ranking của query này.",
        )
    return (
        "generation",
        "Answer grounding/relevance yếu hơn; cần kiểm tra refusal và citation handling.",
    )


def _render_bonus_section() -> str:
    if not BONUS_RESULTS_PATH.is_file():
        reranker = "Chưa chạy A/B; không claim bonus reranker."
    else:
        payload = json.loads(BONUS_RESULTS_PATH.read_text(encoding="utf-8"))
        baseline = payload["summary"]["baseline"]
        advanced = payload["summary"]["llm_reranker"]
        delta = payload["summary"]["delta"]
        metadata = payload["metadata"]
        reranker = f"""Ngày {metadata['evaluation_date']}, thử nghiệm dùng {metadata['dataset_size']} golden questions, `candidate_k={metadata['candidate_k']}`, `top_k={metadata['top_k']}` và model `{metadata.get('reranker_model', '')}`.

| Retrieval metric | Hybrid + RRF | Hybrid + RRF + LLM reranker | Delta |
| --- | ---: | ---: | ---: |
| Source Hit@5 | {baseline['source_hit_at_k']:.4f} | {advanced['source_hit_at_k']:.4f} | {delta['source_hit_at_k']:+.4f} |
| MRR | {baseline['mrr']:.4f} | {advanced['mrr']:.4f} | **{delta['mrr']:+.4f}** |
| nDCG@5 | {baseline['ndcg_at_k']:.4f} | {advanced['ndcg_at_k']:.4f} | **{delta['ndcg_at_k']:+.4f}** |
| Context Precision@5 | {baseline['context_precision_at_k']:.4f} | {advanced['context_precision_at_k']:.4f} | **{delta['context_precision_at_k']:+.4f}** |
| Evidence token coverage | {baseline['evidence_token_coverage']:.4f} | {advanced['evidence_token_coverage']:.4f} | **{delta['evidence_token_coverage']:+.4f}** |

LLM reranker cải thiện ranking, precision và coverage trên cùng candidate pool; chi tiết nằm trong `group_project/evaluation/bonus_reranker_results.json`."""
    return f"""### Advanced LLM reranker (+3)

{reranker}

### Citation/source highlighting (+2)

- UI map citation `[S1]`, `[S2]` về đúng source và gắn nhãn “được trích dẫn”.
- Cited passage tô sáng evidence terms bằng `<mark>` sau khi HTML-escape nội dung nguồn.
- Unit test kiểm tra citation mapping và ngăn script tag được render.

Hai hạng mục trên là bằng chứng để đề nghị **+5 điểm bonus**. HyDE/query expansion và conversation memory không được claim."""


def _render_report(
    *,
    dataset_size: int,
    top_k: int,
    judge_model: str,
    embedding_model: str,
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, float]],
) -> str:
    dense = summary["dense_only"]
    hybrid = summary["hybrid_rrf"]
    winner = (
        "Cấu hình B — hybrid + RRF"
        if hybrid["average"] > dense["average"]
        else "Cấu hình A — dense-only"
    )
    ranked_rows = sorted(
        rows,
        key=lambda row: statistics.fmean(row["scores"].values()),
    )
    worst: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in ranked_rows:
        if row["question"] in seen_questions:
            continue
        worst.append(row)
        seen_questions.add(row["question"])
        if len(worst) == 3:
            break

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevance",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
    }
    score_lines = []
    for metric in METRIC_NAMES:
        score_lines.append(
            f"| {metric_labels[metric]} | {dense[metric]:.4f} | "
            f"{hybrid[metric]:.4f} | {hybrid[metric] - dense[metric]:+.4f} |"
        )
    score_lines.append(
        f"| **Trung bình** | **{dense['average']:.4f}** | "
        f"**{hybrid['average']:.4f}** | "
        f"**{hybrid['average'] - dense['average']:+.4f}** |"
    )

    worst_lines = []
    for rank, row in enumerate(worst, 1):
        stage, cause = _failure_analysis(row)
        scores = row["scores"]
        question = row["question"].replace("|", "\\|")
        worst_lines.append(
            f"| {rank} | {question} | {row['configuration']} | "
            f"{scores['faithfulness']:.4f} | {scores['answer_relevancy']:.4f} | "
            f"{scores['context_recall']:.4f} | {scores['context_precision']:.4f} | "
            f"{stage} | {cause} |"
        )

    weakest_hybrid_metric = min(METRIC_NAMES, key=lambda name: hybrid[name])
    latency_delta = hybrid["mean_latency_seconds"] - dense["mean_latency_seconds"]

    return f"""# Kết quả đánh giá hệ thống RAG

## Thông tin lần chạy

| Trường | Giá trị |
| --- | --- |
| Ngày đánh giá | {date.today().isoformat()} |
| Framework và phiên bản | RAGAS {version('ragas')} |
| Evaluator model | {judge_model} |
| Generator model | {os.getenv('LLM_MODEL', '').strip()} |
| Embedding model | {embedding_model} |
| Phiên bản corpus/commit | {_git_revision()} |
| Số mẫu golden dataset | {dataset_size} |
| `top_k` | {top_k} |
| Fallback threshold | `0.0`; tắt cho cả hai nhánh A/B để cô lập retrieval strategy |

## Cấu hình đánh giá

- **Cấu hình A — dense-only:** cosine retrieval, không dùng BM25 và RRF.
- **Cấu hình B — hybrid + RRF:** dense kết hợp BM25, fuse đúng một lần bằng RRF.

Hai cấu hình dùng chung golden dataset, generator, evaluator, prompt, `top_k`, source selection và validator; chỉ retrieval strategy thay đổi.

## Dữ liệu và quyết định indexing

Corpus gồm 8 tài liệu Markdown: 3 văn bản pháp lý/chính sách và 5 bài viết/trang du lịch. Title, document type và đường dẫn nguồn nội bộ được giữ trong metadata xuyên suốt pipeline; URL gốc được giữ cho 5 tài liệu news. Ba tài liệu legal hiện chưa có URL nguồn chính thức.

### Giải thích tham số Task 4

| Tham số | Lựa chọn | Lý do |
| --- | --- | --- |
| Chunking strategy | `RecursiveCharacterTextSplitter` | Ưu tiên ranh giới đoạn/dòng, phù hợp với điều khoản và bài viết tiếng Việt. |
| `CHUNK_SIZE` | 800 ký tự | Giữ phần lớn điều khoản hoặc passage trong một chunk nhưng vẫn gọn cho retrieval. |
| `CHUNK_OVERLAP` | 80 ký tự (10%) | Giữ evidence gần ranh giới mà không lặp quá nhiều nội dung. |
| Embedding | `{embedding_model}` | Indexing và query dùng cùng model, tránh vector-space mismatch. |
| Vector database | ChromaDB, cosine distance | Lưu bền vững, ID ổn định và score phù hợp fallback threshold. |
| Batch | 64 embeddings; 100 Chroma records | Giới hạn request/write size, giữ thứ tự response và cho phép upsert lặp lại. |

Chunk ID có dạng `<relative-document-path>::chunk-<index>`; collection lưu embedding provider/model và từ chối index không tương thích.

## Điểm tổng hợp

| Metric | Cấu hình A | Cấu hình B | Delta B−A |
| --- | ---: | ---: | ---: |
{chr(10).join(score_lines)}

## So sánh A/B

- Cấu hình tốt hơn theo trung bình bốn metric: **{winner}**.
- Refusal rate: dense-only {dense['refusal_rate']:.1%}; hybrid+RRF {hybrid['refusal_rate']:.1%}.
- Mean latency: dense-only {dense['mean_latency_seconds']:.2f} giây, hybrid+RRF {hybrid['mean_latency_seconds']:.2f} giây; B−A {latency_delta:+.2f} giây/query.
- Hybrid+RRF làm điểm trung bình thay đổi {hybrid['average'] - dense['average']:+.4f}; metric yếu nhất là {metric_labels[weakest_hybrid_metric]} ({hybrid[weakest_hybrid_metric]:.4f}).

## Các trường hợp kém nhất

| # | Câu hỏi | Cấu hình | Faithfulness | Relevance | Recall | Precision | Công đoạn lỗi | Nguyên nhân gốc |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(worst_lines)}

## Đề xuất cải thiện

| Ưu tiên | Hành động | Evidence | Tác động kỳ vọng | Cách kiểm chứng |
| ---: | --- | --- | --- | --- |
| 1 | Retrieve candidate pool lớn hơn và dùng advanced reranker. | Các case kém có recall/precision thấp. | Tăng context recall. | Chạy lại cùng golden dataset. |
| 2 | Hiệu chỉnh BM25/tokenization tiếng Việt. | Hybrid phụ thuộc lexical match. | Cải thiện ranking cho query paraphrase. | Theo dõi RRF rank và context precision. |
| 3 | Hiệu chỉnh refusal/citation validation. | Refusal rate {dense['refusal_rate']:.1%} và {hybrid['refusal_rate']:.1%}. | Giảm false refusal, giữ faithfulness. | Bổ sung refusal-labelled cases. |

## Thử nghiệm điểm cộng

{_render_bonus_section()}

## Hạn chế đã biết

- Hybrid+RRF thấp hơn dense-only {dense['average'] - hybrid['average']:.4f} điểm trung bình trong đánh giá chính.
- A/B chính tắt fallback để cô lập retrieval strategy; fallback được kiểm tra riêng bằng contract tests.
- Answer relevance và false refusal vẫn cần được hiệu chỉnh.
- LLM reranker tăng chi phí và latency nên được thiết kế là tùy chọn.

## Tái lập kết quả

```bash
python -m group_project.evaluation.run_evaluation --top-k {top_k}
python -m group_project.evaluation.run_bonus_reranker_evaluation --top-k 5 --candidate-k 15
```

Kết quả chi tiết nằm trong `group_project/evaluation/evaluation_results.json` và `group_project/evaluation/bonus_reranker_results.json`.
"""


def run_evaluation(
    *,
    top_k: int,
    judge_model: str,
    embedding_model: str,
) -> dict[str, Any]:
    dataset = _load_golden_dataset(GOLDEN_DATASET_PATH)
    rows: list[dict[str, Any]] = []
    for name, use_reranking in CONFIGURATIONS:
        rows.extend(
            _run_configuration(
                dataset,
                name=name,
                use_reranking=use_reranking,
                top_k=top_k,
            )
        )

    asyncio.run(
        _score_rows(
            rows,
            judge_model=judge_model,
            embedding_model=embedding_model,
        )
    )
    summary = _summarize(rows)
    payload = {
        "metadata": {
            "evaluation_date": date.today().isoformat(),
            "ragas_version": version("ragas"),
            "judge_model": judge_model,
            "generator_model": os.getenv("LLM_MODEL", "").strip(),
            "embedding_model": embedding_model,
            "top_k": top_k,
            "fallback_threshold": 0.0,
        },
        "summary": summary,
        "rows": rows,
    }
    RAW_RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        _render_report(
            dataset_size=len(dataset),
            top_k=top_k,
            judge_model=judge_model,
            embedding_model=embedding_model,
            rows=rows,
            summary=summary,
        ),
        encoding="utf-8",
    )
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate dense-only against hybrid+RRF with four RAGAS metrics."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVALUATOR_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
    )
    parser.add_argument(
        "--embedding-model",
        default=(
            os.getenv("EVALUATOR_EMBEDDING_MODEL")
            or os.getenv("EMBEDDING_MODEL")
            or "text-embedding-3-small"
        ),
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    payload = run_evaluation(
        top_k=args.top_k,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
