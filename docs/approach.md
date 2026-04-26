# Carbon Crunch — OCR Receipt Extraction
## Approach Documentation

**Author:** Tirupathamma Guntru
**Date:** April 2026
**Assignment:** AI-OCR Shortlisting

---

## 1. Approach

The system is a five-stage pipeline, deliberately built from
well-understood components rather than a single end-to-end neural
network. Each stage is independently testable and the contract between
stages is a plain Python dictionary, which keeps the system easy to
debug and extend.

```
Image  →  Preprocess  →  OCR (EasyOCR)  →  Field extraction
                                             →  Confidence scoring  →  JSON
```

**Stage 1 — Preprocessing (`src/preprocess.py`).**
Receipts arrive in unpredictable conditions: skewed phone photos,
washed-out lighting, motion blur, low-resolution scans. The
preprocessor applies, in order: (a) an upscale to 1500 px height for
small inputs (more pixels per character helps OCR), (b) CLAHE on the
L-channel of LAB color space to flatten uneven lighting without
distorting colors, (c) edge-preserving non-local-means denoising, and
(d) deskew based on `cv2.minAreaRect` over the binarized image. Each
step is conservative — over-processing destroys text edges, which OCR
cares about more than humans do.

**Stage 2 — OCR (`src/ocr.py`).**
EasyOCR is the engine of choice. It's well-supported on Windows,
returns per-line bounding boxes and confidence scores natively, and
needs no system-level binaries (unlike Tesseract's `tesseract.exe`
dependency). The wrapper normalizes output into a fixed schema so a
future swap to PaddleOCR or LayoutLM only changes one file.

**Stage 3 — Field extraction (`src/extractor.py`).**
Rather than a heavyweight ML model, this stage uses **regex + positional
heuristics**, which for receipts are surprisingly accurate and fully
deterministic. Specifically:
- *Date* — regex over multiple international formats (DD/MM/YYYY,
  YYYY-MM-DD, "12 Apr 2026", etc.).
- *Store name* — first non-numeric, sufficiently-long line near the
  top of the receipt; bonus confidence if it's the very first line.
- *Total* — keyword-anchored search ("Grand Total", "Total", "Amount")
  with a fallback to the largest plausible currency value if no
  keyword matches. Priority order: Grand Total > Net Total > Total >
  Amount > Sum.
- *Items* — lines matching `<text> <price>` pattern, excluding any
  line containing a summary keyword (tax, subtotal, change, etc.).

**Stage 4 — Confidence scoring (`src/confidence.py`).**
Each field's final confidence combines three independent signals:
```
final = ocr_conf × validation_factor × heuristic_factor
```
- `ocr_conf` — what EasyOCR thinks of the text quality.
- `validation_factor` — does the value parse cleanly? Dates are tested
  against 15+ formats; currencies must fall in the [0.01, 1,000,000]
  range; store names must have ≥3 alpha characters.
- `heuristic_factor` — already baked into the extractor's raw output:
  e.g., a total found via "Grand Total" anchor scores 0.92×, a
  fallback-largest-number total scores 0.55×.

Anything below 0.7 is added to the receipt's `low_confidence_flags`
list — these are exactly the rows a human reviewer should look at
first.

**Stage 5 — Financial summary (`src/summary.py`).**
Aggregates per-receipt outputs into total spend, transaction count,
average transaction, per-store breakdown, max/min transaction, and the
count of receipts flagged for review.

---

## 2. Tools used

| Tool             | Purpose                                                      |
| ---------------- | ------------------------------------------------------------ |
| **EasyOCR**      | Text recognition with per-line confidence                    |
| **OpenCV**       | Denoising, contrast enhancement, deskewing                   |
| **NumPy**        | Image and statistics arrays                                  |
| **pandas**       | Tabular analysis in the demo notebook                        |
| **matplotlib**   | Visualizations (preview images, confidence histograms)       |
| **Jupyter**      | Interactive demo                                             |
| **argparse**     | CLI for `main.py`                                            |

EasyOCR was selected over Tesseract (poor confidence-per-line API,
brittle Windows install) and PaddleOCR (PaddlePaddle dependency is
unreliable on Windows). EasyOCR is pure-Python with PyTorch under the
hood and works out-of-the-box.

A heavyweight LLM (GPT-4 Vision, Gemini) was deliberately avoided: it
would add latency, cost, and a network dependency for a problem where
deterministic heuristics already work well. The system can be extended
to call an LLM only for *very low* confidence receipts as a fallback —
this is noted in the Improvements section.

---

## 3. Challenges faced

**Currency-format ambiguity.** Receipts use `1,234.56` (US/IN) and
`1.234,56` (EU) interchangeably, and sometimes mix them within the
same receipt. The parser uses position of the rightmost separator to
decide which is the decimal mark, with a fallback rule for clean
thousands-separated integers (`1,234,567`).

**Total amount ambiguity.** Many receipts have multiple total-like
lines — *Subtotal*, *Total*, *Grand Total*, *Amount Tendered*. The
extractor uses a priority order (Grand Total > Net Total > Total >
others) and skips anything containing tax/cash/change keywords.

**Skew detection failure mode.** `cv2.minAreaRect` can return spurious
angles on receipts with very few text pixels. Mitigated by (a)
requiring ≥50 foreground pixels before attempting deskew, and (b)
skipping rotation entirely for sub-degree skew (which would only add
resampling artifacts).

**Item line vs. noise line.** Many "noisy" lines look like
`<word> <number>` (e.g., a phone number, a receipt code, a date with a
suffix). Mitigated by requiring at least 3 alpha characters in the
name, length ≤80, and rejecting any line whose lowercase form contains
a summary keyword.

**Cold-start latency.** EasyOCR loads ~64 MB of model weights on first
use. Mitigated by lazy initialization in `src/ocr.py` so it only
happens once per process.

---

## 4. Improvements (with more time)

1. **Fine-tune EasyOCR** on a receipt-specific dataset. Off-the-shelf
   weights are trained on general printed text — domain-specific
   fine-tuning typically lifts accuracy by 5–10 points.
2. **Layout-aware extraction** with **LayoutLM** for receipts where
   line-items span multiple visual columns (e.g., quantity ⨯ unit-price
   = subtotal layouts).
3. **Multilingual support** — current pipeline assumes English; adding
   Hindi / Telugu / Tamil would significantly improve coverage in the
   Indian retail market. EasyOCR supports 80+ languages — just a
   `Reader(['en','hi','te'])` change.
4. **LLM fallback for low-confidence receipts.** Use a small LLM
   (Gemini Flash, GPT-4o-mini) *only* on receipts whose pipeline
   confidence is below 0.5 — keeps costs minimal while improving
   coverage on the hardest 5–10% of inputs.
5. **Active learning loop.** Capture reviewer corrections on flagged
   receipts and feed them back into a fine-tuning dataset, so the
   system improves over time without manual rule changes.
6. **Visual line-grouping.** Use OCR bounding-box geometry to group
   line-items that wrap onto two visual lines (very common in narrow
   thermal-printed receipts).
7. **Per-region confidence calibration.** The `0.7` threshold was
   chosen as a sensible default; in production this should be
   calibrated per-deployment based on observed reviewer agreement.

---

## 5. Reproducibility

```bash
git clone <repo>
cd Intern
python -m venv venv && source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# place dataset in data/receipts/
python main.py
```

All outputs are deterministic (no LLM calls, no randomness in the
pipeline) — the same inputs always produce the same JSON.
