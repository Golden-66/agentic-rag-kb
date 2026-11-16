"""检索评测: 计算 Hit Rate@K / MRR / Recall@K。

数据集 (jsonl), 每行:
  {"question": "...", "expected": ["应出现在命中上下文中的关键片段", ...]}
判定规则: 重排后的第 i 个上下文只要包含任意一个 expected 片段, 即视为在第 i 位命中。

用法:
  python eval/run.py --dataset eval/dataset.jsonl --namespace default --k 5
"""
import argparse
import json
import os
import sys

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    ap.add_argument("--api", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}
    with open(args.dataset, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if not items:
        sys.exit("数据集为空")

    hits = rr = recall = 0.0
    for it in items:
        r = httpx.post(
            f"{args.api}/api/v1/retrieve",
            json={"query": it["question"], "namespace": args.namespace},
            headers=headers,
            timeout=300,
        )
        r.raise_for_status()
        texts = [c["text"] for c in r.json()["contexts"]][: args.k]
        expected = it["expected"]
        first = next((i for i, t in enumerate(texts, 1) if any(e in t for e in expected)), None)
        hits += first is not None
        rr += 1 / first if first else 0
        recall += sum(any(e in t for t in texts) for e in expected) / len(expected)
        print(f"{'HIT ' if first else 'MISS'} rank={first}  {it['question']}")

    n = len(items)
    print(f"\nsamples={n}  Hit@{args.k}={hits / n:.3f}  MRR={rr / n:.3f}  Recall@{args.k}={recall / n:.3f}")


if __name__ == "__main__":
    main()
