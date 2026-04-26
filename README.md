# Carbon Crunch — OCR Receipt Extraction

> Shortlisting Assignment — submitted by **Tirupathamma Guntru**
> (tirupathammaguntru@gmail.com)

End-to-end pipeline that turns raw receipt images into clean, structured,
**confidence-aware** JSON, ready for downstream financial applications.

---

## What it does

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw image   │ →  │ Preprocess   │ →  │ EasyOCR      │ →  │ Extract +    │
│  (jpg/png)   │    │ (denoise,    │    │ (per-line    │    │ score fields │
│              │    │  contrast,   │    │  confidence) │    │ (regex +     │
│              │    │  deskew)     │    │              │    │  heuristics) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                                            ┌──────────────────────▼─────────┐
                                            │   per-receipt JSON +           │
                                            │   portfolio financial summary  │
                                            └────────────────────────────────┘
```

For each receipt the system extracts:

* **Store / vendor name**
* **Date of transaction**
* **Line items + individual prices**
* **Total amount**

Every extracted field carries a **confidence score (0–1)** combining:
1. The OCR engine's own confidence,
2. Format validation (date parsing, currency-range plausibility),
3. Heuristic anchors (proximity to keywords like "Total" / "Grand Total").

Fields scoring below 0.7 are **automatically flagged for human review**.

---

## Repository layout

```
Intern/
├── README.md                    ← you are here
├── requirements.txt
├── main.py                      ← end-to-end CLI runner
├── notebook.ipynb               ← demo with visualizations
├── data/
│   └── receipts/                ← drop the dataset images here
├── outputs/
│   ├── json/                    ← one .json per receipt
│   ├── preview/                 ← before/after preprocessing samples
│   └── summary.json             ← portfolio-level aggregations
├── src/
│   ├── preprocess.py            ← noise / contrast / deskew
│   ├── ocr.py                   ← EasyOCR wrapper, normalized output schema
│   ├── extractor.py             ← field extractors (regex + heuristics)
│   ├── confidence.py            ← multi-signal confidence scoring
│   └── summary.py               ← financial aggregation
└── docs/
    └── approach.md              ← 1–2 page write-up
```

---

## Quick start

### 1. Install
```bash
git clone <this-repo>
cd Intern
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

> First run downloads EasyOCR model weights (~64 MB). Subsequent runs are fast.

### 2. Add data
Download the receipt dataset from the assignment link and unzip into:
```
data/receipts/
```

### 3. Run
```bash
python main.py
```

Outputs:
* `outputs/json/<image>.json` — structured extraction per receipt
* `outputs/summary.json` — total spend, per-store breakdown, etc.
* `outputs/preview/*.png` — first 5 before/after preprocessing samples

### 4. Explore in the notebook
```bash
jupyter notebook notebook.ipynb
```
The notebook walks through every stage on a single receipt and then on
the whole dataset, with visualizations of confidence distribution and
per-store spend.

---

## Output schema

```jsonc
{
  "file": "receipt_001.jpg",
  "store_name":   { "value": "Walmart Supercenter", "confidence": 0.92 },
  "date":         { "value": "12/04/2026",          "confidence": 0.95 },
  "items": [
    {
      "name":  "Whole Wheat Bread",
      "price": { "value": "45.00", "confidence": 0.81 }
    },
    {
      "name":  "Toned Milk 1L",
      "price": { "value": "62.50", "confidence": 0.78 }
    }
  ],
  "total_amount": { "value": "187.50", "confidence": 0.96 },
  "low_confidence_flags": [],
  "metadata": {
    "ocr_lines_detected": 23,
    "average_ocr_confidence": 0.87,
    "preprocessing": {
      "blur_score": 412.5,
      "brightness": 168.3,
      "skew_angle": -1.2,
      "original_shape":  [1242, 928, 3],
      "processed_shape": [1500, 1121, 3]
    }
  },
  "raw_ocr_text": "Walmart Supercenter\n... full OCR dump ..."
}
```

`outputs/summary.json`:
```jsonc
{
  "total_transactions_processed": 25,
  "successful_extractions": 24,
  "failed_extractions": 1,
  "receipts_with_total_amount": 23,
  "low_confidence_receipt_count": 4,
  "total_spend": 12834.75,
  "average_transaction": 558.03,
  "spend_per_store": {
    "Walmart Supercenter": { "total": 4502.10, "transactions": 6, "average": 750.35 },
    "Big Bazaar":          { "total": 2887.40, "transactions": 4, "average": 721.85 }
  },
  "max_single_transaction": 1840.00,
  "min_single_transaction": 12.50
}
```

---

## How confidence is calculated

```
final_confidence  =  ocr_confidence  ×  validation_factor  ×  heuristic_factor
```

| Signal              | What it captures                                                                |
| ------------------- | ------------------------------------------------------------------------------- |
| `ocr_confidence`    | EasyOCR's per-line probability that the text was read correctly                 |
| `validation_factor` | Does the value parse as a valid date? Is the currency in a plausible range?     |
| `heuristic_factor`  | Was the value found via a strong anchor (`"Grand Total"` line)? Bonus / penalty |

Anything **below 0.7** is added to the receipt's `low_confidence_flags`
list — these are the rows a human reviewer should look at first.

---

## Tools used

| Layer            | Library                                  |
| ---------------- | ---------------------------------------- |
| OCR              | **EasyOCR** (per-line confidence, GPU-optional) |
| Image processing | **OpenCV** (denoise, CLAHE, deskew)       |
| Data wrangling   | **pandas**, **numpy**                    |
| Notebook / viz   | **matplotlib**, **Jupyter**              |
| CLI              | **argparse**, standard library           |

EasyOCR was chosen over Tesseract because it ships per-line confidence
scores out-of-the-box (Tesseract only offers per-word at lower granularity)
and over PaddleOCR because PaddleOCR's PaddlePaddle dependency is fragile
on Windows.

---

## Edge cases handled

* **Blank / unreadable images** → flagged with an explicit `error` field
  instead of crashing the pipeline.
* **Skewed receipts** → auto-deskewed (angle detection via `minAreaRect`).
* **Low-light images** → CLAHE on the L-channel of LAB color space fixes
  uneven lighting without over-saturating.
* **Tiny images** → upscaled to 1500 px height before OCR.
* **Currency in `1,234.56`, `1.234,56`, or `1234`** → unified parser.
* **Missing total** → falls back to the largest plausible currency value
  in the receipt, with reduced confidence.
* **No items detected** → returned as empty list rather than null.
* **Non-English / mixed scripts** → still attempts English OCR; flags
  low confidence for review.

---

## Re-running on your own data

```bash
python main.py --input my_receipts/ --output my_outputs/
python main.py --no-preview          # skip preview images for speed
python main.py --max-previews 10     # save more preview comparisons
```

---

## Future improvements

* Fine-tune the OCR model on a receipt-specific corpus (currently
  general-purpose EasyOCR weights).
* Layout-aware extraction with **LayoutLM** for receipts where line-items
  span multiple visual columns.
* Multilingual support (Hindi / Telugu / Tamil — important for the Indian
  receipt market).
* Active learning loop: feed reviewer corrections of low-confidence
  fields back into the heuristics.
* Optional **LLM fallback** for the very-low-confidence receipts only,
  keeping cost minimal.

---

## License

Submitted as part of the Carbon Crunch shortlisting assignment.
