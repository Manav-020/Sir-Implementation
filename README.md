# EEG Emotion Recognition: Subject-Specific Personalization and Explainability

## Research question and current finding

**Does subject-specific calibration make EEG emotion explanations more faithful and stable to the model’s actual decision process while maintaining predictive performance?**

This README documents the uploaded, executed **`DEAP_384_Correlation_LogReg(1).ipynb`**. It describes the 384-feature implementation and its saved outputs. It does not describe the later 128-feature redesign.

**Main finding:** Personalization improves predictive performance in the separate unseen-trial, stimulus-only evaluation. However, this run does **not** show a statistically significant improvement in explanation faithfulness in any of the four evaluated conditions. Explanation stability is already near its ceiling, and its tiny changes do not establish a practically meaningful improvement. The combined research hypothesis is therefore **not established by this implementation**.

Results below were read from the notebook’s saved outputs, not reproduced by rerunning the original EEG here. The notebook introduction says outputs are cleared, but this uploaded copy contains executed outputs and a complete results table; that introductory statement is stale.

## 1. Data and experiment settings

| Setting | Implemented value |
| --- | --- |
| Dataset | User-supplied DEAP copy; official signal equivalence unverified |
| People / trials | 32 people × 40 trials = 1,280 trials |
| Stored signal shape per person | 40 trials × 40 channels × 8,064 samples |
| EEG inputs | First 32 channels; peripheral channels excluded |
| Sampling rate | 128 Hz |
| Baseline | First 3 seconds |
| Retained stimulus | Stimulus time 5–60 seconds |
| Two retained halves | 5–32.5 seconds and 32.5–60 seconds |
| Features per segment | 384 = 32 channels × 4 frequency bands × 3 metrics |
| Correlation threshold | Remove candidates with absolute Pearson correlation > 0.90 to an already retained feature |
| Classifier | Logistic Regression, `C=1.0`, balanced class weights, maximum 5,000 iterations |
| Calibration weighting | 10× trial-level weighting relative to a source trial |
| Split seed | `[42]`: one seed in the recorded run |
| Explanation | Analytic interventional linear SHAP in High-Valence log-odds space |
| Faithfulness setting | Top 10 features versus random 10-feature masking; 30 random repetitions |
| Stability setting | Gaussian noise SD 0.05 in standardized feature units; 10 repetitions |
| Inference | Paired subject-level Wilcoxon tests, 5,000 bootstrap resamples, Holm correction across 20 comparisons |

**384 refers to engineered features, not EEG channels.** The initial feature-correlation matrix is 384 × 384, subject to removal of constant features.

## 2. Step-by-step explanation

### Step 1 — Configure the input and output folders

The notebook uses:

```python
DATA_DIR = Path(r'D:\coding\python\HCI RESEARCH PROJECT\imlementation\data_preprocessed_python')
OUT = Path('deap_professor_updated_results')
```

`imlementation` follows the supplied folder spelling. The output path is relative to the current notebook working directory. The saved run reports `D:\coding\python\HCI RESEARCH PROJECT\deap_professor_updated_results`.

The same output directory is reused on reruns, so files may be overwritten. Archive a completed run or change `OUT` before running a different configuration.

### Step 2 — Audit the files and recover reference labels

The loader checks for `s01.dat` through `s32.dat` and verifies signal/label shapes. Each stored label is compared with the embedded reference from the supplied `participant_ratings.xls`.

Reference matching uses **person ID + video/Experiment ID**, not presentation order. The spreadsheet’s `Trial` field represents presentation order; it is stored for verification but is not used for chronological splitting in this version.

All Valence labels used for fitting come from the matched metadata, including stored values that were inside the valid range. This is important because a transformed label can look valid while still being wrong. Values are not clipped or guessed, and original `.dat` files remain unchanged.

Dominance and Liking act as alignment anchors. If they do not match the reference, fitting stops. Reference labels must also be finite and within 1–9.

The saved outputs show:

| Dimension | Stored entries matching metadata | Reference entries outside 1–9 |
| --- | ---: | ---: |
| Valence | 841 / 1,280 | 0 |
| Arousal | 841 / 1,280 | 0 |
| Dominance | 1,280 / 1,280 | 0 |
| Liking | 1,280 / 1,280 | 0 |

Thus 439 stored Valence entries and 439 stored Arousal entries do not directly match the reference. Counts of values matching `9 − reference` can overlap direct matches at particular values; they should not automatically be interpreted as counts of distinct corrupted rows.

The original audit reported 163 out-of-range stored ratings, including 110 Valence ratings. The zero-invalid output here describes the **reference ratings used after replacement**, not proof that all original stored labels were valid.

The embedded spreadsheet SHA256 is:

```text
512191a9db0814ac3042e601e5ed4a2d8a578223678d19b74b29825df9a7ea9a
```

This identifies the supplied metadata reference; it does not certify the EEG signals.

### Step 3 — Create the binary Valence task

- **Low Valence:** rating ≤ 5.
- **High Valence:** rating > 5.

Saved corrected counts are **572 Low** and **708 High**. The previously reported 79%/21% class distribution does not describe these corrected labels. Class proportions can still vary substantially by person and split.

### Step 4 — Extract baseline and stimulus-half features

Each trial contains 3 seconds of baseline followed by 60 seconds of stimulus. “Start from the sixth second” means skip the first five seconds of **stimulus**, after the baseline.

| Segment | Stored sample slice | Duration |
| --- | --- | ---: |
| Baseline | `0:384` | 3 seconds |
| First retained stimulus half | `1024:4544` | 27.5 seconds |
| Second retained stimulus half | `4544:8064` | 27.5 seconds |

Features are extracted independently for each segment. This prevents the feature-extraction filter itself from operating jointly across the two retained halves. It does not remove the fact that both halves belong to the same trial.

### Step 5 — Calculate 384 features per segment

The four bands are Theta 4–8 Hz, Alpha 8–13 Hz, Beta 13–30 Hz and Gamma 30–45 Hz. PSD integration uses half-open intervals: lower bound included, upper bound excluded.

For every channel and band, calculate:

1. **Absolute power:** Welch PSD integrated over the band.
2. **Relative power:** band power divided by total 4–45 Hz power.
3. **Differential entropy:** Gaussian entropy approximation from variance after fourth-order band-pass filtering with zero-phase filtering.

There are `32 × 4 × 3 = 384` values. Delta-band features are not included in this version.

### Step 6 — Compare two feature modes

| Mode | Inputs |
| --- | --- |
| `stimulus` | Extracted features from the stimulus segment |
| `baseline_relative` | Each stimulus segment contextualized using that trial’s 3-second baseline |

For baseline-relative inputs, absolute power becomes the log stimulus/baseline power ratio. Relative power and differential entropy become stimulus-minus-baseline differences.

The baseline is per trial, not a separate long-term personal resting profile. Both halves of a trial use the same baseline reference.

### Step 7 — Train a source model using the other 31 people

For target person S, train the general model on the other 31 people. Each source person contributes 40 trials, with two segment rows per trial:

```text
31 × 40 × 2 = 2,480 source segment rows
```

Each source segment receives sample weight 0.5 so that a source trial contributes total sample weight 1 before class weighting. These rows still represent only 1,240 distinct source trials; the two halves are not independent trials.

### Step 8 — Fit preprocessing on source data only

The notebook:

1. Fits a median imputer on source features.
2. Removes constant features.
3. Computes source-training Pearson correlations.
4. Greedily removes a candidate if its absolute correlation with an already retained feature exceeds 0.90.
5. Saves retained and removed feature lists.
6. Computes rank, condition number and VIF diagnostics before and after filtering.
7. Fits a scaler on retained source features.

The resulting preprocessing is frozen for general/personalized comparisons. Target calibration and test data do not fit the imputer, filter or scaler.

**Unlike the later redesign, this upload includes multicollinearity diagnostics.** Their actual numeric outcomes are saved in per-subject CSVs and are not displayed in the uploaded notebook. It is therefore not possible to conclude from this file alone that multicollinearity was resolved.

### Step 9 — Evaluate two personalization protocols

#### A. `temporal_halves`: adaptation within the same trials

- Calibration: first retained halves of all 40 target trials.
- Test: second retained halves of those same 40 target trials.
- Each calibration segment receives sample weight 10.

The sample ranges are disjoint, but the trial identities and trial labels overlap. This asks whether labeled earlier segments help classify later segments from already observed trials.

It does **not** establish transfer to separate target trials. The `(trial, half)` disjointness assertion passes even though trial IDs overlap.

#### B. `unseen_trials`: adaptation to separate target trials

- Randomly split the target person’s 40 trials into 20 calibration trials and 20 test trials.
- Stratify by label when both classes have at least two trials.
- Use both halves of calibration trials for fitting.
- Use both halves of test trials for evaluation.
- Each calibration segment receives sample weight 5, making total calibration trial weight 10.

The calibration and test trial IDs are disjoint. The split uses random seed 42 and is **not chronological**, even though presentation order exists in the embedded metadata.

**Calibration and unseen testing are both performed on the held-out target person.** The target person changes across all 32 LOSO evaluations.

### Step 10 — Refit a personalized Logistic Regression model

Combine the source segments with weighted target calibration segments, then fit a new Logistic Regression model. This is weighted supervised refitting, not reinforcement learning or incremental parameter adaptation.

The general model is trained without target labels. Both models use the same source-derived preprocessing and fixed `C=1.0`. However, `class_weight='balanced'` is recomputed for each fitting set, and total fitting weight is not normalized between fits. Differences therefore include changes in class weighting and regularization scale, not solely the addition of correctly labeled personal examples.

### Step 11 — Measure prediction at trial level

For unseen trials, average the probabilities of the two test halves before classifying the trial at probability 0.5. Thus 40 test segment rows represent 20 test trials.

For temporal halves, each of the 40 trials has one test segment.

The notebook records accuracy, balanced accuracy, Macro-F1, ROC-AUC and confusion-matrix counts. Balanced accuracy and ROC-AUC are undefined if the test labels contain only one class.

### Step 12 — Compute linear SHAP and explanation metrics

The analytic contribution for a standardized feature is:

```text
SHAP = coefficient × (feature value − mean source feature value)
```

The notebook verifies that contributions plus expected log odds reconstruct the model’s decision function. This is interventional linear SHAP for High-Valence log odds; it does not require the `shap` package.

**Faithfulness:** rank features by absolute SHAP; replace the top 10 with standardized source medians; measure the probability drop for the model’s originally predicted class. Subtract the average drop after masking 10 randomly chosen features over 30 repetitions:

```text
faithfulness = top-10 probability drop − mean random-10 probability drop
```

Higher values mean top-ranked masking was more disruptive than random masking under this intervention. Negative values are possible. This is not a general percentage of explanation correctness. Absolute SHAP rankings can include evidence opposing the predicted class, so removing important features does not necessarily reduce its probability.

**Stability:** add Gaussian noise with SD 0.05 in standardized feature space, recalculate SHAP, and measure cosine similarity with the original vector. Average over 10 repetitions. This tests local input-noise sensitivity, not retraining or calibration-resampling stability.

The code assigns cosine 1 when both vectors are effectively zero; this can inflate apparent stability in degenerate cases. XAI metrics are averaged over segments, whereas predictive metrics use trial-level probabilities. Cross-model top-10 overlap is also recorded, but is not a primary significance-tested endpoint.

### Step 13 — Compare paired subject outcomes

For each mode/protocol/metric, compute personalized-minus-general differences for the same people. If multiple seeds are configured, average them within person first. This recorded run uses only one seed.

The notebook uses:

- Mean paired difference and 95% subject-bootstrap interval from 5,000 resamples.
- Two-sided paired Wilcoxon tests.
- Holm adjustment across all 20 mode/protocol/metric comparisons.

Report corrected p-values for significance. Bootstrap intervals are not adjusted for all 20 comparisons; an interval excluding zero can coexist with an adjusted p-value above 0.05. Shared LOSO training data also limits simple independent-subject interpretations.

## 3. Recorded results

All comparisons below contain 32 valid subject pairs. Values are rounded from the saved table. Delta is personalized minus general. “Holm p” is the multiple-comparison-adjusted p-value.

### Separate unseen target trials

#### stimulus

| Metric | General | Personalized | Delta | 95% CI for delta | Holm p |
| --- | ---: | ---: | ---: | --- | ---: |
| balanced_accuracy | 0.498338 | 0.622130 | +0.123793 | [+0.070975, +0.175520] | 0.00228931 |
| macro_f1 | 0.366567 | 0.613782 | +0.247215 | [+0.187296, +0.301464] | 4.11657e-05 |
| roc_auc | 0.507079 | 0.702401 | +0.195322 | [+0.112704, +0.286644] | 0.00282041 |
| faithfulness | 0.190700 | 0.163257 | -0.027444 | [-0.108181, +0.052066] | 1 |
| stability | 0.997992 | 0.998094 | +0.000101 | [+0.000012, +0.000198] | 0.49194 |

#### baseline_relative

| Metric | General | Personalized | Delta | 95% CI for delta | Holm p |
| --- | ---: | ---: | ---: | --- | ---: |
| balanced_accuracy | 0.473922 | 0.545878 | +0.071955 | [+0.020010, +0.127139] | 0.0705335 |
| macro_f1 | 0.449558 | 0.528631 | +0.079073 | [+0.027664, +0.134336] | 0.0425478 |
| roc_auc | 0.473759 | 0.561081 | +0.087322 | [+0.040237, +0.137158] | 0.0139317 |
| faithfulness | 0.119324 | 0.131453 | +0.012129 | [-0.020682, +0.046158] | 1 |
| stability | 0.997484 | 0.997546 | +0.000062 | [+0.000009, +0.000112] | 0.0425478 |

### Temporal halves of the same trials — within-trial diagnostic

#### stimulus

| Metric | General | Personalized | Delta | 95% CI for delta | Holm p |
| --- | ---: | ---: | ---: | --- | ---: |
| balanced_accuracy | 0.497830 | 0.690257 | +0.192426 | [+0.156308, +0.228357] | 1.67638e-08 |
| macro_f1 | 0.369313 | 0.685737 | +0.316424 | [+0.273881, +0.357798] | 9.31323e-09 |
| roc_auc | 0.503757 | 0.739865 | +0.236108 | [+0.182067, +0.292564] | 1.58702e-05 |
| faithfulness | 0.194442 | 0.133656 | -0.060786 | [-0.149480, +0.018658] | 0.693689 |
| stability | 0.997967 | 0.998201 | +0.000234 | [+0.000133, +0.000340] | 0.00210448 |

#### baseline_relative

| Metric | General | Personalized | Delta | 95% CI for delta | Holm p |
| --- | ---: | ---: | ---: | --- | ---: |
| balanced_accuracy | 0.488168 | 0.877299 | +0.389131 | [+0.350109, +0.426519] | 1.35124e-05 |
| macro_f1 | 0.466119 | 0.875707 | +0.409588 | [+0.369075, +0.448248] | 9.31323e-09 |
| roc_auc | 0.478984 | 0.949708 | +0.470723 | [+0.433459, +0.507743] | 1.35124e-05 |
| faithfulness | 0.123848 | 0.155632 | +0.031783 | [-0.000974, +0.066224] | 0.534556 |
| stability | 0.997565 | 0.997692 | +0.000127 | [+0.000061, +0.000202] | 0.0105938 |

## 4. Detailed interpretation and research outcome

### 4.1 Personalization helps unseen-trial prediction, especially with stimulus-only features

For stimulus-only inputs on separate unseen trials:

- Balanced accuracy increases from **0.4983 to 0.6221**, an improvement of **12.38 percentage points**; Holm p = **0.00229**.
- Macro-F1 increases from **0.3666 to 0.6138**; Holm p = **0.0000412**.
- ROC-AUC increases from **0.5071 to 0.7024**; Holm p = **0.00282**.

These saved results support a predictive personalization benefit in this protocol. An AUC around 0.70 still represents imperfect discrimination, and the general model is near chance. Predictive gains should not be described as solving EEG emotion recognition or proving explanation quality.

For baseline-relative unseen trials, Macro-F1 and ROC-AUC improve significantly after correction. Balanced accuracy increases by about 7.20 percentage points, but its Holm p = 0.0705 exceeds 0.05. Its unadjusted bootstrap interval excludes zero; this does not override the adjusted significance test.

### 4.2 Faithfulness does not show a significant improvement

Across all four conditions, the adjusted faithfulness p-values exceed 0.05 and all corresponding difference intervals include zero.

The most relevant separate-trial comparisons are:

- **Stimulus:** 0.190700 → 0.163257; delta −0.027444; 95% CI [−0.108181, +0.052066]; Holm p = 1.0.
- **Baseline-relative:** 0.119324 → 0.131453; delta +0.012129; 95% CI [−0.020682, +0.046158]; Holm p = 1.0.

The stimulus-only point estimate actually decreases despite improved prediction. That is a useful finding: better prediction does not automatically produce a larger deletion-based faithfulness score.

Do not report “personalization significantly improved faithfulness” from this notebook. Earlier claims based on different labels or protocols are not supported by these saved results.

### 4.3 Stability is near a ceiling

All reported means are approximately 0.997–0.998. Some tiny changes are significant after Holm correction, particularly for baseline-relative inputs, but the effects are only around 0.00006–0.00023.

For separate unseen trials:

- Stimulus stability increases by **0.000101**, with Holm p = **0.49194**: no corrected significance.
- Baseline-relative stability increases by **0.000062**, with Holm p = **0.04255**: statistically significant but extremely small.

This metric shows that a linear attribution vector barely changes under the chosen small feature noise. It does not show robust consistency across calibration choices, training samples or recording sessions. A ceiling result is a limitation of the measurement as well as a reported outcome.

### 4.4 Baseline-relative features do not improve the main predictive result

On unseen trials, personalized stimulus-only balanced accuracy is **0.6221**, versus **0.5459** for baseline-relative inputs. The AUC comparison is **0.7024** versus **0.5611**.

These descriptive comparisons suggest that this particular baseline-relative formulation is not a useful default. The notebook does not run a paired significance test directly comparing feature modes, so it does not establish that one mode is significantly better than the other. It also does not rule out other resting-context methods.

### 4.5 Temporal-half scores need a different interpretation

Baseline-relative temporal halves yield balanced accuracy **0.8773** and AUC **0.9497** after personalization. These impressive scores concern later portions of the very trials whose earlier portions and labels were used for calibration.

Shared trial identity, recording context, label and baseline can explain part of the large apparent benefit. These numbers are valid descriptions of the implemented within-trial task, but should not be headline evidence of generalization to new trials or sessions.

### 4.6 Answer to the research question

| Component | Outcome supported by this run |
| --- | --- |
| Improved predictive performance | Supported in the stimulus-only separate-trial protocol |
| More faithful explanations | Not demonstrated by the current faithfulness metric |
| More stable explanations | Tiny and condition-dependent changes under a near-ceiling local-noise metric; practical improvement not established |
| Predictive equivalence / noninferiority | Not formally tested; the notebook defines no preservation margin |
| Overall combined hypothesis | Not established |

A defensible conclusion is:

> Subject-specific weighted calibration improved unseen-trial predictive performance, particularly with stimulus-only features. However, no statistically significant faithfulness improvement was observed, and stability remained near its ceiling with very small changes. Therefore, these results support predictive adaptation but do not establish that personalization produces more faithful and meaningfully more stable explanations. The much stronger temporal-half outcomes should be interpreted as within-trial adaptation rather than separate-trial generalization.

Failure to demonstrate the explanation hypothesis is a valid research outcome. Do not change the reported conclusion to match the original expectation.

## 5. Limitations

1. **Same-trial dependence in temporal-half evaluation.** Nonoverlapping samples do not mean independent trials. The `(trial, half)` assertion cannot detect shared trial identity. Use separate-trial results for claims about transfer to unseen trials.
2. **Shared baseline context.** Both halves use the same 3-second trial baseline in baseline-relative mode. This creates shared context and can contribute to within-trial dependence.
3. **Short baseline estimates.** Three seconds provide limited data for stable band-power and entropy estimation. Log ratios and differences can amplify noisy baseline estimates. This implementation does not use a separate 4-second resting segment.
4. **Faithfulness is conditional on the chosen intervention.** Feature masking to medians can create unrealistic EEG feature combinations, particularly with residual correlations. Probability changes also depend on confidence and saturation.
5. **Exact linear SHAP is not improved computationally by personalization.** Under the interventional linear assumptions, attribution additivity is already exact. A different deletion score can reflect a changed decision rule, concentration or scale rather than a more accurate explanation algorithm.
6. **Mismatch between attribution and deletion targets.** SHAP explains High-Valence log odds, while deletion uses each model’s originally predicted-class probability. The models may predict different classes for the same example. Absolute attribution rankings can also include opposing evidence.
7. **Stability ceiling and zero-vector handling.** Small standardized Gaussian noise yields similarity near 1 for this linear setup. Assigning two zero vectors similarity 1 can further obscure degenerate explanations.
8. **No retraining stability assessment.** The noise metric keeps the model fixed. It does not evaluate variation across calibration resamples, random splits, hyperparameter settings or feature-selection refits.
9. **One random split seed.** The actual recorded run uses seed 42 only. Per-person outcomes can depend on the 20/20 split. This version does not evaluate chronological earlier-to-later deployment.
10. **Unverified unofficial signals.** Shape checks and matching Dominance/Liking labels do not prove signal authenticity or EEG-to-video alignment. Original-file hashes document the files used but do not establish official equivalence.
11. **Correlated segment rows.** Weighting source halves by 0.5 changes contribution to the objective, but does not make them independent observations. The source sample size is 1,240 distinct trials, not 2,480 independent trials.
12. **Redundant and skewed features.** Raw absolute power, relative power and differential entropy overlap. Pearson filtering and regularization do not remove all dependencies or guarantee stable coefficients. Actual VIF/rank outputs must be reviewed before claiming success against multicollinearity.
13. **Fixed hyperparameters and weighting confounds.** `C=1.0` and calibration weight 10 are not selected by nested source validation. Balanced class factors are recomputed for each fit, while total sample weight changes; the comparison does not isolate personal information alone.
14. **Missing controls.** There is no majority-class, target-only or shuffled-calibration-label comparison. The experiment cannot cleanly attribute gains to correctly matched personal labels rather than weighting, class proportions or target-distribution information.
15. **Weak general baseline.** General ROC-AUC is near chance. Cross-subject variability, features and possible data issues remain possible explanations. Explainability gains alone would not establish a useful emotion classifier.
16. **No formal preservation test.** A nonsignificant prediction difference would not demonstrate equivalence. A practical noninferiority margin must be justified in a suitable follow-up protocol.
17. **Limited participant count and LOSO dependence.** There are 32 people, and models share source training subjects. This limits simple independent-subject statistical interpretations and broad population claims.
18. **Reporting limits.** The notebook does not display full per-person metrics, confusion matrices, retained feature counts or VIF values. These require the exported CSVs. The default paired plots show baseline-relative mode only; they do not visualize every table condition.
19. **No human evaluation.** Computational metrics do not establish that people find the explanations clear, trustworthy or useful in an HCI application.
20. **Reused output folder and stale introductory text.** Reruns can overwrite artifacts. The introductory “outputs cleared” statement should be updated before sharing this executed notebook.

## 6. How to run

1. Open `DEAP_384_Correlation_LogReg(1).ipynb` in Jupyter or VS Code with a Python kernel.
2. Confirm `DATA_DIR` contains `s01.dat` through `s32.dat`.
3. Install dependencies if required:

   ```bash
   python -m pip install numpy pandas scipy scikit-learn matplotlib joblib ipython jupyter
   ```

4. Confirm `CORRECTIONS_CSV=None`: this implementation uses the embedded metadata reference. A separate Excel reader is not needed.
5. Archive old results or change `OUT` if preserving previous runs.
6. Select **Restart Kernel → Run All**.
7. Stop and investigate if shape, anchor, reference-label, finite-feature or convergence checks fail. Do not bypass them to obtain scores.
8. Inspect the printed label counts and completed-subject messages before interpreting the summary.
9. Review `paired_summary.csv`, `subject_metrics.csv`, `split_manifest.csv` and diagnostics along with the saved figures.

The uploaded execution records Python 3.11.13, NumPy 2.4.0, SciPy 1.15.2 and scikit-learn 1.8.0. Other environments can produce small numerical differences. Load only trusted pickle/joblib data and model files.

## 7. Outputs

All paths below are relative to `deap_professor_updated_results/`.

| Output | Purpose |
| --- | --- |
| `stored_vs_metadata_labels.csv` | Original/reference rating comparisons and transformation checks |
| `metadata_alignment_anchors.csv` | Dominance/Liking anchor checks |
| `data_manifest.csv` | Dataset file sizes and SHA256 hashes |
| `verified_valence_labels.csv` | Reference labels used for fitting, including presentation order |
| `label_audit.csv`, `invalid_ratings_only.csv`, `label_summary.csv` | Validation of reference ratings used by the experiment |
| `trial_metadata.csv` | Person, video/trial ID, corrected Valence and binary target |
| `segment_features.npz` | Baseline, first-half and second-half 384-feature arrays |
| `subject_metrics.csv` | Model outcomes by person, mode, protocol and seed |
| `trial_predictions.csv` | Trial-level labels and averaged probabilities |
| `split_manifest.csv` | Exact source/calibration/test segment membership |
| `paired_summary.csv` | Paired mean differences, intervals and raw/adjusted p-values |
| `metrics_checkpoint.csv` | Intermediate accumulated metrics |
| `configuration.json` | Run settings, reference provenance and package versions |
| `paired_*.png` | Baseline-relative paired outcome plots for both protocols |
| `<mode>/subject_XX/training_correlation.csv` | Source-training feature correlation matrix |
| `<mode>/subject_XX/removed_features.csv`, `retained_features.csv` | Feature filtering decisions |
| `<mode>/subject_XX/vif_before.csv`, `vif_after.csv`, `collinearity_summary.csv` | Multicollinearity diagnostics |
| `<mode>/subject_01/correlation_heatmap.png` | Example correlation heatmap |
| `<mode>/subject_XX/*_xai.csv` | Per-segment faithfulness and stability measurements |
| `<mode>/subject_XX/*_shap.npz` | SHAP values, retained names, trial IDs and half IDs |
| `<mode>/subject_XX/*.joblib` | Fitted models, preprocessing and explanation context |

## 8. Recommended follow-up

Use the separate unseen-trial protocol as the principal evaluation and retain temporal halves only as a clearly labeled diagnostic. For a stronger follow-up, specify a chronological calibration/test split, training-only hyperparameter selection, target-only and shuffled-label controls, and refit-based stability before viewing new results.

If full linear SHAP remains the explanation method, separate its exactness audit from a clearly defined compact-explanation faithfulness question. Any nonlinear model extension should retain fair data partitions and explicitly state the explanation method and evaluation assumptions. Neither switching classifiers nor adding resting context guarantees that the hypothesis will become positive.

**Final outcome of this uploaded implementation:** predictive personalization is promising, but improved faithfulness and practically meaningful explanation stability have not been established.
