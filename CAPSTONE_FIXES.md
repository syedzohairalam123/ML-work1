# Capstone Notebook — Fix Log

This file documents every bug found in `work/notebooks/capstone.ipynb` and the
fix applied, so anyone (human or AI) picking this up cold can understand what
was wrong, why, and what changed — without re-deriving it from scratch.

**Validation method:** the notebook was executed end-to-end, top to bottom,
via a real Jupyter kernel (`jupyter nbconvert --execute`), not just read.
Two validation passes were run after the fixes:

1. **Fast-mode pass** (small synthetic dataset, 2 CV folds, 2 Optuna trials) —
   completed with **0 errors and 0 warnings** across all 24 code cells, and
   produced all 6 CSV/JSON artifacts, all 9 figures, and all 3 ML-12 text
   deliverables correctly.
2. **Near-full-scale pass** (real `N_CLIENTS=45`, real 5-fold CV, 5 Optuna
   trials — trial count only reduced to keep runtime reasonable in the
   sandbox) — run to confirm nothing scale-dependent breaks.

The real Hugging Face data path (cell 5, when `HF_TOKEN` is set) could not be
executed in this environment — no token, no network access to
`huggingface.co`. Its column names (`client_hash_id`, `content_hash_id`,
`gsc_impressions`, `gsc_clicks`, `gsc_avg_position`, `report_date`) and table
name (`fact_content_daily_performance`) were cross-checked against
`notebooks/03_working_with_the_full_release.ipynb` (the vetted starter
notebook for this exact workflow) and match exactly, so this path is very
likely fine — but it has not been executed against live data. Please run it
once with your real `HF_TOKEN` and report back if anything doesn't match; I
was not able to verify this part directly.

**UPDATE (verified against live data):** the real Hugging Face data path (cell 5)
has now been executed successfully with a real `HF_TOKEN`. The exact feature
query from cell 5 ran against
`fact_content_daily_performance/month=2026-03` and returned **116,113 entities
across 44 clients** (9.84M raw rows, 331,437 entities, 55 clients before the
`HAVING` filter), with all expected columns present and only trivial NaNs
(67 `position_volatility`, 1 `engagement_proxy` — both handled by the existing
`fillna`). The live path is confirmed working.

---

## Why it wouldn't run at all

**1. No install cell — the very first code cell crashed on a fresh runtime.**
`duckdb`, `shap`, `huggingface_hub`, `optuna`, and `lime` are not part of
Colab's default image, and the notebook had zero `pip install` cells anywhere.
`import shap` and `import duckdb` are hard, unguarded imports in the first
code cell, so on any fresh Colab runtime (or fresh local environment) the
notebook failed immediately with `ModuleNotFoundError` before running a
single line of actual logic. This is almost certainly the error you were
hitting.
**Fix:** added a new first code cell — `%pip install -q duckdb
huggingface_hub shap optuna lime seaborn` — and added the same packages to
`requirements.txt` (which was also missing `seaborn`, `shap`, `optuna`,
`lime`).

**2. f-strings that only work on Python 3.12+.**
`print(f"\n{"="*70}")` reuses the outer `"` as an inner quote inside the same
f-string — only legal from Python 3.12 onward (PEP 701). Colab may or may not
be on 3.12 yet; this was a silent version trap either way.
**Fix:** switched the outer quotes to single quotes: `f'\n{"="*70}'`, which
is valid on every Python version and behaves identically.

---

## Bugs in the SHAP section (Section 8)

**3. `shap_model.shap_values(X_test)` — called directly on the sklearn
model.** `shap_values()` doesn't exist on a plain `GradientBoostingClassifier`
— it lives on a SHAP *explainer* object that wraps the model. This raised
`AttributeError: 'GradientBoostingClassifier' object has no attribute
'shap_values'` on every run.
**Fix:** `explainer = shap.TreeExplainer(shap_model)`, then
`explainer.shap_values(X_test)`.

**4. Section 8b referenced variables that no longer existed.**
An earlier patch (`repair_capstone_shap_playbook.py`, in
`scripts/archive_old_capstone_patch_attempts/`) renamed the SHAP output
variables in Section 8a from `shap_vals_class1`/`X_shap` to
`shap_matrix`/`X_test` to fix a row-count bug, but never updated Section 8b's
beeswarm/dependence plot cell to match — so it threw `NameError` on both
names.
**Fix:** Section 8b now uses `shap_matrix` / `X_test`, matching Section 8a.

**5. `shap.summary_plot(..., ax=ax)` — `ax` isn't a valid parameter in the
installed SHAP version (0.52).** This only surfaced after fixing #3 and #4.
**Fix:** `plt.sca(ax)` right before the call, so `summary_plot` draws onto
the correct subplot via pyplot's current-axis state instead of a direct `ax=`
argument (`shap.dependence_plot` right after it *does* support `ax=`, so
that call was left as-is).

---

## Bugs in the Action Playbook section (Section 10)

This section had five interlocking bugs, all from the same root cause —
columns being renamed or added in one place without the rest of the section
being updated to match. It was rewritten as one coherent block rather than
patched line by line, to avoid leaving another half-consistent state:

- **`historical_ctr` was referenced in five places but never computed
  anywhere** (a pre-flight check, a merge loop, the action-assignment
  function's required-fields list, the display table, and the final
  required-columns check) — it was always going to raise `ValueError`. It's
  meant to be the already-computed `smoothed_ctr`, just under a
  playbook-friendly name. **Fix:** `df["historical_ctr"] =
  df["smoothed_ctr"]`, added once, early in the section.
- **The pre-flight check validated the wrong DataFrame at the wrong time** —
  it checked `X_test.columns` for columns that actually live on
  `test_results` (a separate copy made in Section 8a), and it checked for
  `priority_score`/`historical_ctr` before either was computed. This
  guaranteed a `ValueError` on every run. **Fix:** split into two checks —
  one right after Section 8a's columns should exist (checking
  `test_results.columns`), and one right before export (checking
  `playbook.columns`, after every column has actually been added).
- **The df → playbook merge loop only copied 3 of the 8 columns it needed.**
  `priority_score`, `decay_probability`, `conformal_set_size`,
  `confidence_factor`, and `structural_score` are only computed on the full
  `df`, but the loop that copies columns across only pulled
  `content_hash_id`, `client_hash_id`, and `historical_ctr` — so the rest
  were missing wherever they were used later (ranking, the action-assignment
  function, the display table, the CSV export). **Fix:** the loop now copies
  all 8.
- **The "validate before export" check ran before the column it checked
  for existed.** It required `reason_code` in `playbook.columns`, but
  `reason_code` wasn't computed until several lines *after* that check.
  **Fix:** moved the check to after `reason_code` is added.
- **A stray, non-f-string error message.** `"Playbook missing required
  columns: {_missing_cols}. "` was missing the `f` prefix, so it would have
  printed the literal text `{_missing_cols}` instead of the actual list.
  Fixed in passing.

**Result after the fix** (from the fast-mode validation run): 154 pages
evaluated, sensible priority scores, a real action distribution (MONITOR
52.6%, CRITICAL: Optimize Meta Titles & Snippets 24.0%, etc.), reason codes
that match the SHAP driver — the section now does what Section 10's markdown
describes.

**6. Duplicate summary cell with the wrong column names (old Section 10b).**
Immediately after the (now-fixed) Section 10 cell, a second cell reprinted
the exact same "Playbook Summary" using `action_label` and
`shap_primary_driver` — neither of which exist; the real columns are
`recommended_action` and `primary_driver`. Since Section 10 already prints
this summary correctly, this cell was 100% redundant on top of being broken.
**Fix:** removed. (Two further cells, Section 11's chart and Section 12's
CSV export, made the same `action_label` mistake — fixed the same way,
`action_label` → `recommended_action`, `shap_primary_driver` →
`primary_driver`.)

**7. The final CSV export was silently failing.** Section 12's export is
wrapped in `try/except`, so the `action_label`/`shap_primary_driver` bug
above (plus the missing `structural_score`, caught in the near-full-scale
run) never raised a visible error — it just printed "Some artifacts could
not be exported" and skipped `final_action_playbook.csv`, your most
important deliverable, without you necessarily noticing. Fixed as part of
#6 and the merge-loop fix above; confirmed the CSV now writes correctly.

---

## Cosmetic but real: stray quote characters in the ML-12 text outputs

The three `work/outputs/*.txt` deliverables (presentation outline, LinkedIn
post, employer summary) were built from f-strings that started and ended
with an extra escaped `""` — almost certainly a leftover from a
triple-quoted string that got typo'd into a single-line f-string. It doesn't
crash anything, but every generated `.txt` file would have literally started
and ended with two stray `"` characters. **Fix:** removed the stray
escape-pairs; verified by actually running the cell and checking the output
files no longer start/end with `""`.

---

## Paper-accuracy fix: the Abstract didn't match the actual methodology

The **Methodology** (Section 5) and **Modeling** (Section 6) sections
correctly describe what's implemented: a structural, peer-relative decay
label (not a temporal one) and a Logistic Regression / Random Forest /
Gradient Boosting stack (not LightGBM/XGBoost/CatBoost — those aren't
imported anywhere in the notebook). The **Abstract** (Section 1), however,
claimed "(LightGBM, XGBoost, CatBoost, Logistic Regression)", "temporal label
engineering comparing a baseline month to a future horizon", and "isotonic
calibration" — none of which match the code (`CalibratedClassifierCV` is
imported but never actually fit; only calibration *diagnostics*, reliability
curves and Brier score, are computed). For a research paper, an abstract
that contradicts its own methodology section is a real problem if anyone
checks. **Fix:** rewrote the abstract to match Sections 5/6/16 exactly —
same model list, same peer-relative label description, "calibration
diagnostics" instead of "isotonic calibration". Also changed the ungrounded
"~150K content entities" to "[N] content entities" to match the paper's own
placeholder convention (`[X.XXX]`) for numbers that depend on your real run.
The top-level notebook title ("...Causal Feature Isolation") was softened
to match — it directly contradicted the paper's own "CANNOT claim causality"
line in Section 2's claims table.

---

## Small cleanup (not bugs, just tidiness)

- Removed the stale `work/notebooks/work/figures/` folder — two leftover
  PNGs from a run that crashed partway through before the fixes.
- Moved the five one-off patch scripts from earlier debugging attempts
  (`fix_all_errors.py`, `fix_capstone_notebook_setup.py`,
  `fix_target_leak.py`, `patch_capstone_fallback.py`,
  `repair_capstone_shap_playbook.py`) into
  `scripts/archive_old_capstone_patch_attempts/` so the working `scripts/`
  folder only contains the graded reference pipeline
  (`01_prepare_features.py` … `05_build_pdf_report.py`, `ml_utils.py`,
  `run_all.py`). Nothing was deleted — they're still in the repo if you want
  them, just out of the way. Safe to delete outright if you'd rather.
- Removed stray `__pycache__/` folders.

## One thing I could not verify, and you should check

`docs/index.html` (the page `submission/paper_url.txt` points at,
`https://syedzohairalam123.github.io/ML-work1/`) looks like a stale
notebook-to-HTML export — its `<title>` is just "capstone" and it doesn't
contain the abstract text at all, which suggests it predates a lot of the
current notebook content. Once you've re-run the fixed notebook with your
real `HF_TOKEN` and have real results, regenerate this page (e.g. Jupyter's
**File → Download as → HTML**, or `jupyter nbconvert --to html
work/notebooks/capstone.ipynb --output docs/index.html`) and redeploy, so
your live paper matches your repo instead of showing old/placeholder
content.
