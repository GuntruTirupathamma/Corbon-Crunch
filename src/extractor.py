"""
Key information extraction from OCR output.

Strategy:
  - Date:        regex over multiple common formats (intl + IN + US + textual)
  - Store name:  first sufficiently-long, alpha-dominant line near the top
                 (positional heuristic + length filter to skip noise)
  - Total:       keyword-anchored search with priority + neighbor-line lookahead
                 + fallback to largest plausible currency value
  - Items:       lines that match "<text> <price>" pattern, excluding any
                 line containing summary/header/footer keywords

All extractors return (value, raw_confidence) tuples. The raw_confidence is
the OCR confidence multiplied by a heuristic factor reflecting how reliable
the extraction logic is for that field.
"""
from __future__ import annotations

import re
from typing import List, Dict, Optional, Tuple
import numpy as np


# ── Patterns ─────────────────────────────────────────────────────────────────
DATE_PATTERNS = [
    # 12/04/2026, 12-04-26, 12.04.2026, 13.12.01
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
    # 2026-04-12, 2026/04/12
    r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b",
    # 12 Apr 2026, 12 April 2026
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\s+\d{2,4})\b",
    # April 12, 2026
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
    r"\d{1,2},?\s+\d{2,4})\b",
]

# Currency pattern — captures the numeric value (with optional symbol prefix)
CURRENCY_RE = re.compile(
    r"(?:(?:₹|Rs\.?|RM|INR|\$|USD|EUR|£|GBP)\s*)?"
    r"(\d{1,3}(?:[,]\d{3})+(?:\.\d{1,2})?|\d+\.\d{1,2}|\d+)"
)

# Decimal-only currency (more reliable signal — receipts almost always show
# totals with 2 decimal places)
DECIMAL_CURRENCY_RE = re.compile(
    r"(?:(?:₹|Rs\.?|RM|INR|\$|USD|EUR|£|GBP)\s*)?"
    r"(\d{1,3}(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})"
)

# Strict price line: "<words> <price-with-decimals>"
PRICE_LINE_RE = re.compile(
    r"^(.+?)\s+(?:₹|Rs\.?|RM|\$)?\s*(\d+(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})\s*[A-Z]?\s*$"
)

TOTAL_KEYWORDS_PRIORITIZED = [
    ("grand total", 0),
    ("grandtotal", 0),
    ("net total", 1),
    ("nett total", 1),
    ("total amount", 1),
    ("amount due", 1),
    ("balance due", 1),
    ("amount payable", 1),
    ("total", 2),
    ("amount", 3),
    ("amt", 3),
    ("net", 3),
    ("sum", 4),
    ("balance", 4),
]

# Lines containing these are NEVER items
ITEM_SKIP_KEYWORDS = [
    "total", "subtotal", "sub total", "sub-total",
    "tax", "gst", "vat", "service",
    "discount", "rounding", "round off",
    "change", "cash", "card", "tendered", "tender",
    "payment", "paid", "tip",
    "thank", "visit", "welcome",
    "invoice", "receipt", "bill no", "bill #",
    "phone", "tel:", "tel ", "email", "address",
    "table", "guest", "server", "cashier",
    "date", "time", "no.",
    "qty", "quantity", "unit",
]

# Lines containing these are usually header/footer noise — skip when finding
# the store name
STORE_NAME_NOISE = [
    "always low",  # "Always Low Prices" tagline
    "supercenter", "open ", "manager",
    "tel:", "phone", "address",
    "thank", "welcome", "receipt",
    "invoice", "bill no", "bill #", "order #",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _parse_amount(s: str) -> Optional[float]:
    """Convert '1,234.56' or '1.234,56' (eu) → 1234.56 float."""
    s = str(s).strip().replace(" ", "")
    if not s:
        return None
    # Heuristic: if both . and , present, the rightmost is the decimal
    if "." in s and "," in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # If exactly groups of 3 digits → thousands separator
        if re.match(r"^\d{1,3}(,\d{3})+$", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    alpha = sum(1 for c in s if c.isalpha())
    return alpha / max(len(s), 1)


def _looks_like_date_fragment(value: str, all_text: str) -> bool:
    """
    True if `value` is just a date fragment (e.g. '13.12' is part of '13.12.01').
    Prevents date prefixes from being mistaken for currency totals.
    """
    if not value:
        return False
    # Search for date-like pattern containing this value
    # e.g. value='13.12' would match in '13.12.01' or '13.12.2026'
    pattern = re.escape(value) + r"[/\-.]\d{2,4}"
    return bool(re.search(pattern, all_text))


def _merge_lines_into_rows(lines: List[Dict], y_tolerance: int = 18) -> List[Dict]:
    """
    Receipts are columnar layouts — OCR often splits 'BANANAS  0.20' into two
    separate detections. Merge detections that share the same horizontal row.

    Returns a list of {text, confidence, y_center} where text is the joined
    row content, sorted left-to-right.
    """
    if not lines:
        return []
    # Group lines by y-coordinate proximity
    sorted_lines = sorted(lines, key=lambda l: l["y_center"])
    rows: List[List[Dict]] = []
    for line in sorted_lines:
        if rows and abs(line["y_center"] - rows[-1][0]["y_center"]) <= y_tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])

    merged = []
    for row in rows:
        # Sort items in this row left-to-right by bbox x
        row_sorted = sorted(row, key=lambda l: l["bbox"][0][0] if l.get("bbox") else 0)
        text = " ".join(l["text"] for l in row_sorted)
        conf = float(np.mean([l["confidence"] for l in row_sorted]))
        y    = float(np.mean([l["y_center"] for l in row_sorted]))
        merged.append({"text": text, "confidence": conf, "y_center": y})
    return merged


# ── Extractors ───────────────────────────────────────────────────────────────
def extract_date(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    """Returns (date_string, confidence) or None."""
    for line in lines:
        for pattern in DATE_PATTERNS:
            m = re.search(pattern, line["text"], re.IGNORECASE)
            if m:
                return m.group(1), line["confidence"] * 0.95
    return None


def extract_store_name(lines: List[Dict]) -> Tuple[str, float]:
    """
    Heuristic: store name is one of the first few alpha-dominant lines,
    long enough to be a real name, not a tagline or address line.
    """
    if not lines:
        return "Unknown", 0.0

    for idx, line in enumerate(lines[:6]):
        text = line["text"].strip()
        text_lower = text.lower()

        # Quality filters
        if len(text) < 3:
            continue
        if _alpha_ratio(text) < 0.5:
            continue
        if any(noise in text_lower for noise in STORE_NAME_NOISE):
            continue
        # Skip if it looks like a date / phone / number
        if re.match(r"^[\d\s/\-.:,()+]+$", text):
            continue
        # Skip very short single-letter combinations
        if sum(1 for c in text if c.isalpha()) < 3:
            continue

        position_bonus = 0.10 if idx == 0 else (0.05 if idx == 1 else 0.0)
        # Boost: ALL CAPS short words at the top are usually brand names
        if text.isupper() and 3 <= len(text) <= 30:
            position_bonus += 0.05
        conf = min(1.0, line["confidence"] * 0.90 + position_bonus)
        return text, conf

    # Fallback — first line, low confidence
    return lines[0]["text"], lines[0]["confidence"] * 0.4


def extract_total(lines: List[Dict]) -> Optional[Tuple[str, float]]:
    """
    Find the total amount.

    Priority:
      1. "Grand Total" keyword line
      2. "Net Total / Amount Due / Balance Due"
      3. "Total" (alone)
      4. Other total-like keywords
      5. Fallback — most-frequent or bottom-half largest plausible currency value
    """
    # Use row-merged lines so 'TOTAL' (left col) and '5.11' (right col) end up
    # together, even when EasyOCR returned them as separate detections.
    merged = _merge_lines_into_rows(lines)
    full_text = " ".join(l["text"] for l in merged)

    candidates = []
    for i, line in enumerate(merged):
        text_lower = line["text"].lower()

        # Skip lines that are clearly NOT totals
        if any(neg in text_lower for neg in
               ["subtotal", "sub total", "sub-total",
                "discount", "rounding", "tax", "gst", "vat"]):
            continue

        for keyword, priority in TOTAL_KEYWORDS_PRIORITIZED:
            if keyword in text_lower:
                # Find decimal price in this row first
                matches = list(DECIMAL_CURRENCY_RE.finditer(line["text"]))
                price_str = None
                if matches:
                    price_str = matches[-1].group(1)
                elif i + 1 < len(merged):
                    next_decimals = list(DECIMAL_CURRENCY_RE.finditer(
                        merged[i + 1]["text"]))
                    if next_decimals:
                        price_str = next_decimals[-1].group(1)

                if price_str:
                    parsed = _parse_amount(price_str)
                    # Reject date fragments masquerading as currency
                    if _looks_like_date_fragment(price_str, full_text):
                        continue
                    if parsed is not None and 0.01 <= parsed < 1_000_000:
                        decimal_bonus = 0.05 if "." in price_str else -0.10
                        candidates.append({
                            "value": f"{parsed:.2f}",
                            "confidence": min(1.0,
                                line["confidence"] * 0.92 + decimal_bonus),
                            "priority": priority,
                        })
                        break

    if candidates:
        candidates.sort(key=lambda c: (c["priority"], -c["confidence"]))
        best = candidates[0]
        return best["value"], best["confidence"]

    # ── Fallback: collect plausible decimal amounts from merged rows ─────
    all_amounts = []
    for line in merged:
        text = line["text"].strip()
        # Skip noise
        if re.search(r"\d{6,}", text):           continue   # SKUs/codes
        if re.search(r"[A-Z]{2,}\d{4,}", text):  continue   # ITC5679...
        if len(re.findall(r"\d", text)) > 12:    continue   # transaction ids

        for m in DECIMAL_CURRENCY_RE.finditer(text):
            raw  = m.group(1)
            parsed = _parse_amount(raw)
            if parsed is None or not (0.01 <= parsed <= 100_000):
                continue
            # Critical: skip date fragments like '13.12' from '13.12.01'
            if _looks_like_date_fragment(raw, full_text):
                continue
            all_amounts.append((parsed, line["confidence"], line["y_center"]))

    if all_amounts:
        from collections import Counter
        counter = Counter(round(v, 2) for v, _, _ in all_amounts)
        most_common_val, freq = counter.most_common(1)[0]
        if freq >= 2:
            conf = max(c for v, c, _ in all_amounts if round(v, 2) == most_common_val)
            return f"{most_common_val:.2f}", conf * 0.7

        # Largest plausible value in the bottom half (totals are at the bottom)
        n = len(merged)
        if n > 1:
            mid_y = merged[n // 2]["y_center"]
            bottom = [(v, c) for v, c, y in all_amounts if y >= mid_y]
            if bottom:
                bottom.sort(key=lambda x: -x[0])
                return f"{bottom[0][0]:.2f}", bottom[0][1] * 0.6

        # Final fallback
        all_amounts.sort(key=lambda x: -x[0])
        v, c, _ = all_amounts[0]
        return f"{v:.2f}", c * 0.45

    return None


def extract_items(lines: List[Dict]) -> List[Dict]:
    """
    Extract line-items.

    A line is treated as an item if:
      - It does NOT contain any summary / header / footer keyword
      - It matches "<text> <decimal-price>" pattern
      - The text portion has at least 3 alpha chars and isn't a code/SKU
    """
    items = []
    for line in lines:
        text = line["text"].strip()
        text_lower = text.lower()

        # Skip summary / header / footer lines
        if any(k in text_lower for k in ITEM_SKIP_KEYWORDS):
            continue

        m = PRICE_LINE_RE.match(text)
        if not m:
            continue

        name = m.group(1).strip()
        price_raw = m.group(2)
        price = _parse_amount(price_raw)

        # Quality filters on the name part
        alpha_count = sum(1 for c in name if c.isalpha())
        if alpha_count < 3 or price is None or price <= 0:
            continue
        if len(name) > 80:
            continue
        # Skip if name is dominated by codes (e.g. "ST# 5748 OP# 00000158 TE# 14 TR#")
        if _alpha_ratio(name) < 0.4:
            continue
        # Skip lines with too many '#' or ':' — usually transaction codes
        if name.count("#") >= 2 or name.count(":") >= 2:
            continue

        items.append({
            "name": name,
            "price": f"{price:.2f}",
            "confidence": line["confidence"] * 0.78,
        })

    return items
