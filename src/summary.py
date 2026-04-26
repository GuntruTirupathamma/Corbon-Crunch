"""
Financial summary generator.

Aggregates structured per-receipt outputs into a portfolio-level summary:

  - Total spend across all receipts
  - Number of transactions processed (and how many failed)
  - Spend per store
  - Average transaction value
  - Count of receipts flagged as low-confidence
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict


def _to_float(amount) -> float | None:
    if amount is None:
        return None
    try:
        return float(str(amount).replace(",", ""))
    except (ValueError, TypeError):
        return None


def generate_summary(results: List[Dict]) -> Dict:
    """
    Build a financial summary from the per-receipt extraction results.

    Args:
        results: list of dicts as produced by main.process_receipt(), each
                 containing structured fields with confidence scores.

    Returns:
        Summary dict (JSON-serializable).
    """
    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    per_store_total = defaultdict(float)
    per_store_count = defaultdict(int)
    valid_amounts = []
    low_conf_count = 0

    for r in successes:
        amt_field = r.get("total_amount", {}) or {}
        amt = _to_float(amt_field.get("value"))
        if amt is None or amt <= 0:
            continue

        store_field = r.get("store_name", {}) or {}
        store = (store_field.get("value") or "Unknown").strip() or "Unknown"

        valid_amounts.append(amt)
        per_store_total[store] += amt
        per_store_count[store] += 1

        if r.get("low_confidence_flags"):
            low_conf_count += 1

    total_spend = sum(valid_amounts)
    avg_txn = total_spend / len(valid_amounts) if valid_amounts else 0.0

    summary = {
        "total_transactions_processed": len(results),
        "successful_extractions": len(successes),
        "failed_extractions": len(failures),
        "receipts_with_total_amount": len(valid_amounts),
        "low_confidence_receipt_count": low_conf_count,
        "total_spend": round(total_spend, 2),
        "average_transaction": round(avg_txn, 2),
        "spend_per_store": {
            store: {
                "total": round(per_store_total[store], 2),
                "transactions": per_store_count[store],
                "average": round(per_store_total[store] / per_store_count[store], 2),
            }
            for store in sorted(per_store_total,
                                key=lambda s: -per_store_total[s])
        },
        "max_single_transaction": round(max(valid_amounts), 2) if valid_amounts else 0.0,
        "min_single_transaction": round(min(valid_amounts), 2) if valid_amounts else 0.0,
    }
    return summary
