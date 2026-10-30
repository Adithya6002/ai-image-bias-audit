# Bias audit: statistical tests for AI image-generation bias

Runs the *required experiments* of Track C (Bias, Fairness and Sociotechnical Auditing) on the
perceived race and gender annotations of images from ChatGPT Images 2.5, Nano Banana 2 and Qwen-Image 3.0.

```bash
uv sync                         # or: pip install -r requirements.txt
uv run python run.py            # writes results/REPORT.md and results/*.csv
uv run pytest                   # sanity tests (includes reproducing the report's chi-square = 16.77)
```

| Path | Contents |
|---|---|
| `data/` | Input workbook `Aggregation.xlsx` plus the real-world statistics and research report docs |
| `bias_audit/data.py` | Workbook loader, benchmarks, the report's earlier coding |
| `bias_audit/stats.py` | Exact binomial, Monte Carlo exact permutation and GOF tests, bootstrap, effect sizes, fragility |
| `bias_audit/analysis.py` | Experiments E1 to E5 and robustness checks R1 to R5 |
| `bias_audit/report.py` | Markdown report renderer |
| `results/` | Generated output: `REPORT.md` is the main read |

H0 throughout: **there is no bias**. Decisions use Holm-Bonferroni correction at alpha = 0.05.
See `results/REPORT.md` for the operational definition of H0 in each test.
