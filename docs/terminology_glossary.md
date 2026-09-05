# Terminology Glossary — Tier A Precision Pass

Use these terms consistently across ALL scripts, comments, figures, and paper
text. This is the single most important, zero-cost fix available: it makes
every claim in the paper accurate as stated, resolving most circularity
objections rhetorically rather than requiring new data.

| Never say | Say instead | Why |
|---|---|---|
| "defect labels" / "defect ground truth" | "thermal-process severity states" / "proxy severity labels" | Labels are derived from TAM/SCR, not measured defects. No external defect ground truth currently exists for this dataset. |
| "defect detection" | "severity assessment" / "risk governance" | We assess process severity from thermal signals, not confirmed physical defects. |
| "physical law" / "physics limit" (re: VED envelope) | "statistical process admissibility envelope" | The ±2σ band is derived from this dataset's own distribution, not an independently published IN718 process window. |
| "physics validation" (re: TAM-SCR check) | "self-consistency check" | The check is fit on layers labeled "Stable" by the same proxy-labeling process it evaluates — internally consistent, not independently validated. |
| "our model achieves the best accuracy" | "our model achieves comparable accuracy while producing structurally coherent uncertainty" | Random Forest has higher raw accuracy. The ordinal model's actual advantage is 100% adjacent confidence sets vs. RF's ~92%. |
| "MIRI predicts risk" / "MIRI is a risk metric" | "MIRI is a diagnostic index summarizing internal governance-pipeline agreement" | MIRI is learned from action_tier, which is itself derived from the same gates — not validated against independent outcomes. |
| "world's first" / "novel paradigm" (unqualified) | "to the best of our knowledge, the first framework that..." | Standard academic hedge; avoids an overclaim reviewers specifically watch for. |
| "100% accuracy" (bare) | "100% accuracy (95% CI: [x, y], n=...)" | Always report the bootstrap CI and sample size alongside any point estimate, especially for small autonomous subsets. |

## One-paragraph scope statement (use near the start of your methods section)

"This study evaluates a decision-governance architecture using thermal-process
severity states derived from in-situ TAM/SCR measurements as a proxy for
manufacturing risk, given the current unavailability of spatially registered
defect ground truth (e.g., segmented XCT porosity data) for this dataset. We
refer to these as severity states throughout, and explicitly distinguish
internal pipeline consistency validation (confirmed) from independent
defect-outcome validation (identified as necessary future work)."
