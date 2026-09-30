"""分析 BEIR Benchmark 中不同类型的检索失败。"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


# 六类结果互斥，并且能够覆盖所有 Query。
CATEGORY_ORDER = (
    "stable_success",
    "reranker_rescue",
    "reranker_harm",
    "reranker_failure",
    "candidate_pool_miss",
    "retrieval_miss",
)


def load_report(
    input_path: str | Path,
) -> dict:
    """读取 beir_runner.py 生成的 JSON 报告。"""

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def classify_record(
    record: dict,
    top_k: int,
) -> str:
    """
    将一条 Query 分类。

    retrieval_miss:
        Gold 连 RRF Top-100 都没有进入。

    candidate_pool_miss:
        Gold 在 Top-100，
        但没进入 Reranker 的 Top-20 candidate pool。

    stable_success:
        RRF Top-K 和 Reranker Top-K 都成功。

    reranker_rescue:
        RRF Top-K 失败，
        Reranker 成功把 Gold 拉回 Top-K。

    reranker_harm:
        RRF Top-K 原本成功，
        Reranker 反而把 Gold 挤出 Top-K。

    reranker_failure:
        Gold 已进入 candidate pool，
        但 RRF 和 Reranker Top-K 都失败。
    """

    candidate = record[
        "candidate_analysis"
    ]

    methods = record[
        "methods"
    ]

    hit_key = f"hit@{top_k}"

    rrf_hit = (
        float(
            methods["rrf"][hit_key]
        )
        > 0
    )

    rerank_hit = (
        float(
            methods["rerank"][hit_key]
        )
        > 0
    )

    rrf_recall_100 = float(
        candidate[
            "rrf_recall@100"
        ]
    )

    gold_in_pool = bool(
        candidate[
            "gold_in_rerank_pool"
        ]
    )

    # 连 Top-100 都没找到。
    if rrf_recall_100 <= 0:
        return "retrieval_miss"

    # Top-100 找到了，
    # 但没有进入 Reranker 的 candidate pool。
    if not gold_in_pool:
        return "candidate_pool_miss"

    if rrf_hit and rerank_hit:
        return "stable_success"

    if rrf_hit and not rerank_hit:
        return "reranker_harm"

    if not rrf_hit and rerank_hit:
        return "reranker_rescue"

    return "reranker_failure"


def safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    """安全计算比例。"""

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def summarize_categories(
    records: list[dict],
    top_k: int,
) -> dict:
    """统计六类 Query 的数量和比例。"""

    categories = [
        classify_record(
            record,
            top_k=top_k,
        )
        for record in records
    ]

    counter = Counter(
        categories
    )

    counts = {
        category: counter.get(
            category,
            0,
        )
        for category in CATEGORY_ORDER
    }

    total = len(
        records
    )

    rates = {
        category: safe_rate(
            count,
            total,
        )
        for category, count
        in counts.items()
    }

    stable_success = counts[
        "stable_success"
    ]

    reranker_rescue = counts[
        "reranker_rescue"
    ]

    reranker_harm = counts[
        "reranker_harm"
    ]

    reranker_failure = counts[
        "reranker_failure"
    ]

    retrieval_miss = counts[
        "retrieval_miss"
    ]

    # RRF Top-K 成功的 Query。
    rrf_hit_queries = (
        stable_success
        + reranker_harm
    )

    # Reranker Top-K 成功的 Query。
    rerank_hit_queries = (
        stable_success
        + reranker_rescue
    )

    # Gold 至少有一个进入 candidate pool。
    gold_in_pool_queries = (
        stable_success
        + reranker_rescue
        + reranker_harm
        + reranker_failure
    )

    # Gold 至少有一个进入 RRF Top-100。
    gold_in_top100_queries = (
        total
        - retrieval_miss
    )

    # RRF Top-K 没找到，
    # 但 Reranker 有机会补救的 Query。
    rescue_opportunities = (
        reranker_rescue
        + reranker_failure
    )

    return {
        "total_queries": total,

        "counts": counts,

        "rates": rates,

        "derived": {
            f"rrf_hit@{top_k}_queries": (
                rrf_hit_queries
            ),

            f"rrf_hit@{top_k}_rate": (
                safe_rate(
                    rrf_hit_queries,
                    total,
                )
            ),

            f"rerank_hit@{top_k}_queries": (
                rerank_hit_queries
            ),

            f"rerank_hit@{top_k}_rate": (
                safe_rate(
                    rerank_hit_queries,
                    total,
                )
            ),

            "gold_in_candidate_pool_queries": (
                gold_in_pool_queries
            ),

            "gold_in_candidate_pool_rate": (
                safe_rate(
                    gold_in_pool_queries,
                    total,
                )
            ),

            "gold_in_rrf_top100_queries": (
                gold_in_top100_queries
            ),

            "gold_in_rrf_top100_rate": (
                safe_rate(
                    gold_in_top100_queries,
                    total,
                )
            ),

            "reranker_rescue_opportunities": (
                rescue_opportunities
            ),

            # 当 RRF Top-K 失败，
            # 但 Gold 已进入 candidate pool 时，
            # Reranker 有多大概率救回来。
            "reranker_rescue_success_rate": (
                safe_rate(
                    reranker_rescue,
                    rescue_opportunities,
                )
            ),

            # RRF 原本成功时，
            # 有多少被 Reranker 弄坏。
            "reranker_harm_rate_given_rrf_hit": (
                safe_rate(
                    reranker_harm,
                    rrf_hit_queries,
                )
            ),

            # 只要 Gold 已经进入 candidate pool，
            # Reranker 最终 Top-K 成功的比例。
            "reranker_success_rate_given_gold_in_pool": (
                safe_rate(
                    rerank_hit_queries,
                    gold_in_pool_queries,
                )
            ),
        },
    }


def build_case_row(
    record: dict,
    category: str,
    top_k: int,
) -> dict:
    """将一条 Query 转换成 CSV 行。"""

    candidate = record[
        "candidate_analysis"
    ]

    methods = record[
        "methods"
    ]

    hit_key = f"hit@{top_k}"

    return {
        "query_id": record[
            "query_id"
        ],

        "query": record[
            "query"
        ],

        "category": category,

        "relevant_chunk_ids": json.dumps(
            record[
                "relevant_chunk_ids"
            ],
            ensure_ascii=False,
        ),

        "rrf_first_relevant_rank": (
            candidate[
                "rrf_first_relevant_rank"
            ]
        ),

        "rrf_relevant_ranks": json.dumps(
            candidate[
                "rrf_relevant_ranks"
            ],
            ensure_ascii=False,
        ),

        "gold_in_rerank_pool": (
            candidate[
                "gold_in_rerank_pool"
            ]
        ),

        f"bm25_hit@{top_k}": (
            methods[
                "bm25"
            ][hit_key]
        ),

        f"dense_hit@{top_k}": (
            methods[
                "dense"
            ][hit_key]
        ),

        f"rrf_hit@{top_k}": (
            methods[
                "rrf"
            ][hit_key]
        ),

        f"rerank_hit@{top_k}": (
            methods[
                "rerank"
            ][hit_key]
        ),

        "rrf_recall@20": (
            candidate[
                "rrf_recall@20"
            ]
        ),

        "rrf_recall@100": (
            candidate[
                "rrf_recall@100"
            ]
        ),
    }


def build_failure_cases(
    records: list[dict],
    top_k: int,
) -> list[dict]:
    """
    生成值得人工检查的 Case。

    stable_success 不写入 CSV，
    其余五类全部保存。
    """

    rows = []

    for record in records:
        category = classify_record(
            record,
            top_k=top_k,
        )

        if category == "stable_success":
            continue

        rows.append(
            build_case_row(
                record=record,
                category=category,
                top_k=top_k,
            )
        )

    return rows


def analyze_report(
    report: dict,
    source_path: str | Path | None = None,
) -> tuple[dict, list[dict]]:
    """分析完整 Benchmark Report。"""

    top_k = int(
        report[
            "configuration"
        ][
            "project_top_k"
        ]
    )

    records = report[
        "records"
    ]

    summary = summarize_categories(
        records=records,
        top_k=top_k,
    )

    analysis_report = {
        "source_report": (
            str(source_path)
            if source_path is not None
            else None
        ),

        "dataset": report.get(
            "dataset"
        ),

        "split": report.get(
            "split"
        ),

        "project_top_k": top_k,

        "retrieval_depth": (
            report[
                "configuration"
            ].get(
                "retrieval_depth"
            )
        ),

        "rerank_candidate_k": (
            report[
                "configuration"
            ].get(
                "rerank_candidate_k"
            )
        ),

        "summary": summary,
    }

    failure_cases = (
        build_failure_cases(
            records=records,
            top_k=top_k,
        )
    )

    return (
        analysis_report,
        failure_cases,
    )


def save_json(
    data: dict,
    output_path: str | Path,
) -> None:
    """保存 Failure Analysis JSON。"""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_failure_csv(
    rows: list[dict],
    output_path: str | Path,
) -> None:
    """保存需要人工检查的 Case。"""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        output_path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames = list(
        rows[0].keys()
    )

    # utf-8-sig 方便 Windows Excel 打开中文。
    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def default_output_paths(
    input_path: str | Path,
) -> tuple[Path, Path]:
    """自动生成两个输出文件名。"""

    input_path = Path(
        input_path
    )

    summary_path = (
        input_path.parent
        / (
            f"{input_path.stem}"
            "_failure_analysis.json"
        )
    )

    csv_path = (
        input_path.parent
        / (
            f"{input_path.stem}"
            "_failure_cases.csv"
        )
    )

    return (
        summary_path,
        csv_path,
    )


def print_summary(
    analysis_report: dict,
) -> None:
    """在终端打印核心分析。"""

    summary = analysis_report[
        "summary"
    ]

    counts = summary[
        "counts"
    ]

    rates = summary[
        "rates"
    ]

    derived = summary[
        "derived"
    ]

    top_k = analysis_report[
        "project_top_k"
    ]

    print()
    print("=" * 72)
    print(
        "Retrieval Failure Analysis"
    )
    print("=" * 72)

    print(
        f"Total queries: "
        f"{summary['total_queries']}"
    )

    print()

    for category in CATEGORY_ORDER:
        print(
            f"{category:<24}"
            f"{counts[category]:>5}"
            f"  "
            f"{rates[category]:>8.2%}"
        )

    print("-" * 72)

    print(
        f"RRF Hit@{top_k}: "
        f"{derived[f'rrf_hit@{top_k}_rate']:.2%}"
    )

    print(
        f"Reranker Hit@{top_k}: "
        f"{derived[f'rerank_hit@{top_k}_rate']:.2%}"
    )

    print(
        "Gold in candidate pool: "
        f"{derived['gold_in_candidate_pool_rate']:.2%}"
    )

    print(
        "Gold in RRF Top-100: "
        f"{derived['gold_in_rrf_top100_rate']:.2%}"
    )

    print(
        "Reranker rescue success: "
        f"{derived['reranker_rescue_success_rate']:.2%}"
    )

    print(
        "Reranker harm given RRF hit: "
        f"{derived['reranker_harm_rate_given_rrf_hit']:.2%}"
    )

    print(
        "Reranker success when Gold in pool: "
        f"{derived['reranker_success_rate_given_gold_in_pool']:.2%}"
    )

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze LocalDoc-Agent "
            "BEIR benchmark failures."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "beir_runner.py 生成的 "
            "完整 JSON 报告。"
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=None,
    )

    parser.add_argument(
        "--csv-output",
        default=None,
    )

    args = parser.parse_args()

    report = load_report(
        args.input
    )

    (
        analysis_report,
        failure_cases,
    ) = analyze_report(
        report=report,
        source_path=args.input,
    )

    (
        default_summary,
        default_csv,
    ) = default_output_paths(
        args.input
    )

    summary_output = (
        Path(
            args.summary_output
        )
        if args.summary_output
        else default_summary
    )

    csv_output = (
        Path(
            args.csv_output
        )
        if args.csv_output
        else default_csv
    )

    save_json(
        analysis_report,
        summary_output,
    )

    save_failure_csv(
        failure_cases,
        csv_output,
    )

    print_summary(
        analysis_report
    )

    print()

    print(
        "[Analysis] JSON saved: "
        f"{summary_output}"
    )

    print(
        "[Analysis] CSV saved: "
        f"{csv_output}"
    )


if __name__ == "__main__":
    main()