# Deep line-by-line walkthrough of `1_metabolite_final.ipynb` and `2_FINAL_metabolite_interpretation.ipynb`

This walkthrough explains the notebooks at the level of: what each cell is doing, what each line or small group of related lines means, what the math means, and what the results imply for the ME/CFS biomarker project.

Because many Python statements span multiple lines, I explain them by exact line ranges rather than pretending each wrapped line is independent. When a single operation is split over several lines, the whole line range is one logical statement.

---

## Big picture before the line-by-line walkthrough

### Notebook 1: `1_metabolite_final.ipynb`
This is the core metabolomics-modeling notebook. It:

1. Loads metadata and metabolomics.
2. Restricts metabolomics to baseline / `tp1` samples.
3. Builds the disease label `y`, where ME/CFS = 1 and Control = 0.
4. Builds the metabolite matrix `X`.
5. Freezes a held-out 20% test set.
6. Performs repeated LASSO feature selection on the training set only.
7. Ranks metabolites by how consistently LASSO selects them.
8. Tests panel sizes like 3, 5, 8, 10, and 15.
9. Chooses a final / tradeoff panel.
10. Trains a logistic regression classifier.
11. Evaluates it with AUC, accuracy, balanced accuracy, confusion matrix, permutation testing, bootstrapping, ROC/PR curves, threshold metrics, and inconclusive-zone exploration.

The key modeling idea: select a sparse, interpretable set of metabolites, then use a regularized logistic regression model to convert standardized metabolite values into a probability-like ME/CFS score.

### Notebook 2: `2_FINAL_metabolite_interpretation.ipynb`
This is the interpretation/downstream notebook. It:

1. Loads the locked 10-metabolite panel from Notebook 1.
2. Loads Notebook 1's score file instead of refitting new 10-metabolite coefficients.
3. Plots score distributions.
4. Checks whether metadata like age/BMI/gender explains the score.
5. Builds multi-omics matrices: metabolomics, Quest labs, immune, metagenomics, metadata.
6. Tests combinations of modalities.
7. Performs LASSO stability selection for a 15-feature multi-omics panel.
8. Compares 10-metabolite score vs 15-feature multi-omics score.
9. Runs robustness/permutation/bootstrap/threshold analyses.

Important distinction: Notebook 2 does create a multi-omics model, but it should not replace the Notebook 1 locked 10-metabolite coefficient model.

---

# Core math/statistics used in both notebooks

## 1. Feature matrix `X` and label vector `y`
The model always has:

- `X`: rows = subjects, columns = features/metabolites/labs/microbiome variables.
- `y`: disease label, where `1 = ME/CFS` and `0 = Control`.

Mathematically, if there are `n` people and `p` features, then:

```text
X has shape n × p
y has length n
```

For Notebook 1, the output shows approximately:

```text
X: 214 subjects × 876 metabolite features
y: 135 ME/CFS, 79 controls
```

## 2. Standard scaling / z-scoring
`StandardScaler()` converts each feature into a z-score:

```text
z = (x - mean) / standard_deviation
```

Meaning:

- `z = 0`: the person is exactly average for that metabolite.
- `z = +1`: one standard deviation above average.
- `z = -1`: one standard deviation below average.

This matters because metabolites can have different units/ranges. Logistic regression coefficients are more interpretable after scaling: a coefficient describes the effect of a one-standard-deviation increase in that feature.

## 3. Logistic regression
The model computes a linear score first:

```text
logit = intercept + b1*z1 + b2*z2 + ... + bk*zk
```

Then it converts that logit into a probability using the sigmoid function:

```text
probability = 1 / (1 + exp(-logit))
```

Interpretation:

- Positive coefficient: higher feature value pushes score toward ME/CFS.
- Negative coefficient: higher feature value pushes score toward Control.
- Larger absolute coefficient: stronger effect in the model, assuming standardized features.

The model output is not a clinical probability in the strict medical sense unless externally calibrated. In these notebooks, it is best treated as a disease-separation score.

## 4. LASSO / L1 logistic regression
The LASSO model uses an L1 penalty. Logistic regression normally fits coefficients to separate cases and controls. LASSO adds a penalty:

```text
loss + λ * sum(|coefficients|)
```

The key behavior is that LASSO can force some coefficients exactly to zero. Any feature with coefficient zero is effectively removed. That makes it useful for feature selection.

In scikit-learn, `C` is inverse regularization strength:

```text
small C = stronger penalty = fewer selected features
large C = weaker penalty = more selected features
```

So the notebooks loop through multiple `C` values to see which features survive across many selection settings.

## 5. Ridge / L2 logistic regression
The final predictive model mostly uses L2 regularization:

```text
loss + λ * sum(coefficients²)
```

L2 usually does not force coefficients to zero. Instead, it shrinks them, reducing overfitting.

So the notebooks use:

- L1/LASSO for selecting features.
- L2/ridge logistic regression for final prediction once the panel is locked.

That is a reasonable workflow.

## 6. Stratified split and StratifiedKFold
`stratify=y` and `StratifiedKFold` preserve the case/control ratio in each split.

This matters because the dataset has more ME/CFS than controls. Without stratification, one split could accidentally contain too few controls, making metrics unstable.

## 7. AUC / ROC AUC
AUC measures ranking quality. It answers:

> If I randomly choose one ME/CFS subject and one control, how often does the model assign the ME/CFS subject a higher score?

So:

- AUC = 0.5: random ranking.
- AUC = 1.0: perfect ranking.
- AUC = 0.86: strong but not perfect separation.

AUC does not require choosing a classification threshold.

## 8. Accuracy vs balanced accuracy
Accuracy:

```text
accuracy = (TP + TN) / total
```

Balanced accuracy:

```text
balanced accuracy = (sensitivity + specificity) / 2
```

Balanced accuracy is better when classes are imbalanced because it gives cases and controls equal weight.

## 9. Confusion matrix terms
For ME/CFS = positive class:

- TP: true positive = ME/CFS correctly called ME/CFS.
- FN: false negative = ME/CFS incorrectly called Control.
- TN: true negative = Control correctly called Control.
- FP: false positive = Control incorrectly called ME/CFS.

Derived metrics:

```text
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
```

Sensitivity is about catching ME/CFS. Specificity is about avoiding false positives.

## 10. Youden's J threshold
The notebooks choose a threshold by maximizing:

```text
J = sensitivity + specificity - 1
```

Since `sensitivity = TPR` and `1 - specificity = FPR`, this is equivalent to:

```text
J = TPR - FPR
```

This chooses the point on the ROC curve with the best balance between sensitivity and specificity.

## 11. Bootstrap confidence interval
Bootstrapping repeatedly resamples the test set with replacement, calculates AUC each time, and uses percentiles of those bootstrapped AUCs as an uncertainty interval.

The notebooks use the 2.5th and 97.5th percentiles, which gives a rough 95% confidence interval.

## 12. Permutation test
A permutation test shuffles the labels. If the model still performs well after labels are shuffled, the result may be random/overfit. If shuffled-label AUCs cluster around 0.5 and the true AUC is much higher, that supports a real signal.

Notebook 1's simple p-value:

```text
p = fraction of permuted AUCs >= real AUC
```

Notebook 2's smoothed p-value:

```text
p = (number of permuted AUCs >= real AUC + 1) / (number of permutations + 1)
```

The `+1` correction prevents reporting exactly zero.

## 13. Mann-Whitney U test
This nonparametric test asks whether one group tends to have higher values than another. It does not assume normal distributions.

In these notebooks, it is used to compare metabolite values or score distributions between Controls and ME/CFS.

## 14. Cohen's d
Cohen's d measures standardized mean difference:

```text
d = (mean_case - mean_control) / pooled_standard_deviation
```

Rough interpretation:

- 0.2: small effect.
- 0.5: medium effect.
- 0.8: large effect.

Sign matters:

- Positive d: higher in ME/CFS.
- Negative d: lower in ME/CFS.

## 15. FDR / Benjamini-Hochberg
When testing many features, raw p-values can produce false positives. FDR correction controls the expected proportion of false discoveries.

The notebooks use Benjamini-Hochberg with:

```python
multipletests(p_values, method="fdr_bh")
```

---

# Notebook 1: `1_metabolite_final.ipynb`

## Cell 1: Clean consensus metabolite panel pipeline

### Lines 1-4
```python
# CLEAN CONSENSUS METABOLITE PANEL PIPELINE
import os, re, warnings
warnings.filterwarnings("ignore")
```
Line 1 labels the notebook section. Lines 3-4 import basic Python utilities and suppress warnings. `os` handles folders/files, `re` handles regular expressions, and `warnings.filterwarnings("ignore")` hides warning messages that might otherwise clutter the notebook.

### Lines 6-13
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
```
These import the scientific Python stack. `numpy` handles arrays/math. `pandas` handles tables. `train_test_split` creates held-out train/test sets. `StratifiedKFold` creates balanced cross-validation folds. `cross_val_score` scores models across folds. `Pipeline` chains preprocessing and modeling safely. `StandardScaler` z-scores features. `LogisticRegression` is the classifier. The metrics evaluate model performance.

### Lines 15-25
```python
BASE = "../.."
META_PATH = f"{BASE}/data/metadata/Metadata_061523.csv"
METAB_PATH = f"{BASE}/data/metabolomics/Metabolomics_maaslined_norun.csv"
RANDOM_STATE = 42
OUT_DIR = f"{BASE}/output/consensus_metabolite_panel_clean"
os.makedirs(OUT_DIR, exist_ok=True)
```
`BASE` says the project root is two folders above the notebook. `META_PATH` points to metadata. `METAB_PATH` points to metabolomics. `RANDOM_STATE = 42` makes random operations reproducible. `OUT_DIR` names the output folder. `os.makedirs(..., exist_ok=True)` creates that folder if it does not already exist.

### Lines 31-32
```python
meta_raw = pd.read_csv(META_PATH)
metab_raw = pd.read_csv(METAB_PATH)
```
These load the metadata and metabolomics CSVs into pandas DataFrames.

### Lines 34-40
```python
metab = (
    metab_raw
    .set_index("Unnamed: 0")
    .T
    .reset_index()
    .rename(columns={"index": "sample_id"})
)
```
The raw metabolomics file has metabolites as rows and samples as columns. Models need subjects as rows and metabolites as columns, so this transposes the table.

- `.set_index("Unnamed: 0")`: uses the metabolite-name column as the row index.
- `.T`: transposes rows and columns.
- `.reset_index()`: turns the old sample IDs into a normal column.
- `.rename(... "sample_id")`: names that column `sample_id`.

After this, each row is one sample/timepoint and each column is one metabolite.

### Lines 42-45
```python
metab["participant_id"] = metab["sample_id"].str.replace(r"_tp\d+$", "", regex=True)
metab["timepoint"] = metab["sample_id"].str.extract(r"_tp(\d+)")
tp1 = metab[metab["timepoint"] == "1"].copy()
```
These parse sample IDs like `SAM000269_tp1`.

- `str.replace(r"_tp\d+$", "")` removes the timepoint suffix, giving `SAM000269`.
- `str.extract(r"_tp(\d+)")` extracts the timepoint number, e.g. `1`.
- `tp1 = ...` keeps only baseline timepoint 1.

The regex math/logic:

- `_tp`: literally matches `_tp`.
- `\d+`: one or more digits.
- `$`: end of string.

So `_tp\d+$` means "a timepoint suffix at the end of the sample ID."

### Lines 47-50
```python
meta_raw["participant_id"] = meta_raw["samples_id"].astype(str)
meta_tp1 = meta_raw[meta_raw["timepoints"] == "tp1"].copy()
df = tp1.merge(meta_tp1, on="participant_id", how="inner")
```
This creates the same `participant_id` column in metadata, keeps only baseline metadata, then merges metabolomics and metadata by participant. `how="inner"` means only participants present in both tables are kept.

### Lines 52-58
```python
y = df["study_ptorhc"].map({"MECFS": 1, "Control": 0})
metadata_cols = list(meta_raw.columns) + ["sample_id", "participant_id", "timepoint"]
feature_cols = [c for c in df.columns if c not in metadata_cols]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
X = X.loc[:, X.notna().all()]
```
This builds the modeling dataset.

- `y`: disease label. ME/CFS becomes 1; Control becomes 0.
- `metadata_cols`: columns that are not metabolites.
- `feature_cols`: all columns not in metadata; intended to be metabolites.
- `apply(pd.to_numeric, errors="coerce")`: converts metabolite values to numbers; invalid values become `NaN`.
- `X.loc[:, X.notna().all()]`: keeps only columns with no missing values.

So `X` is the clean metabolite matrix.

### Lines 60-61
```python
print("Full X:", X.shape)
print("Class counts:", y.value_counts().to_dict())
```
This prints dataset size and label counts. Your output showed `Full X: (214, 876)` and class counts `{1: 135, 0: 79}`.

### Lines 67-75
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)
print("Train:", X_train.shape, y_train.value_counts().to_dict())
print("Test:", X_test.shape, y_test.value_counts().to_dict())
```
This freezes an 80/20 train/test split. `stratify=y` preserves the class ratio. `random_state=42` makes the exact split reproducible. The test set should be touched only once after model/panel decisions are made.

### Lines 82-86
```python
C_GRID = [0.01, 0.03, 0.06, 0.1, 0.3]
N_REPEATS = 20
N_SPLITS = 5
counts = pd.Series(0, index=X_train.columns, dtype=int)
```
These set up stability selection.

- `C_GRID`: different LASSO strengths.
- `N_REPEATS = 20`: repeat cross-validation 20 times.
- `N_SPLITS = 5`: each repeat uses 5 folds.
- `counts`: starts a counter for every metabolite.

Total feature-selection runs are `20 × 5 × 5 = 500`.

### Lines 88-114
```python
for repeat in range(N_REPEATS):
    cv = StratifiedKFold(...)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        for C in C_GRID:
            model = Pipeline([... StandardScaler(), LogisticRegression(penalty="l1", ... C=C) ...])
            model.fit(X_tr, y_tr)
            coefs = model.named_steps["lasso"].coef_[0]
            selected = X_train.columns[coefs != 0]
            counts[selected] += 1
```
This is the core feature-selection engine.

For each repeat, it creates a new stratified 5-fold split of the training set. For each fold, it trains LASSO logistic regression on the fold's training portion only. For each `C`, it checks which metabolites have nonzero coefficients. Every time a metabolite is selected, its count increases by 1.

Important: this selection happens on `X_train`, not the held-out test set. That is good and protects the final test set.

### Lines 116-124
```python
total_runs = N_REPEATS * N_SPLITS * len(C_GRID)
consensus_df = pd.DataFrame({...}).sort_values(...).reset_index(drop=True)
consensus_df["rank"] = consensus_df.index + 1
```
This turns the counts into a ranked table.

- `total_runs = 500`.
- `frequency = count / total_runs`.
- Sorting by count ranks the most stable metabolites first.
- `rank` gives 1, 2, 3, etc.

### Lines 126-129
```python
print("===== TOP 40 CONSENSUS METABOLITES =====")
display(consensus_df.head(40))
consensus_df.to_csv(...)
```
This displays and saves the top selected metabolites.

### Lines 135-138
```python
ordered_features = consensus_df["metabolite"].tolist()
panel_sizes = [3, 5, 8, 10, 15]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```
This creates panel candidates using the consensus ranking. For example, the 10-feature panel is the top 10 metabolites from `consensus_df`. A 5-fold CV object is created for panel evaluation.

### Lines 140-169
```python
results = []
for k in panel_sizes:
    feats = ordered_features[:k]
    model = Pipeline([... StandardScaler(), LogisticRegression(penalty="l2") ...])
    aucs = cross_val_score(model, X_train[feats], y_train, cv=cv, scoring="roc_auc")
    results.append({...})
```
For each panel size, this evaluates a final-style L2 logistic regression using only those top `k` features. It uses cross-validation on the training set. It records the mean and standard deviation of AUC.

Why L2 here? LASSO already selected the features. The final prediction model should be stable and not drop features arbitrarily, so L2 shrinkage is a reasonable choice.

### Lines 171-176
```python
results_df = pd.DataFrame(results)
display(results_df[["panel_size", "mean_train_cv_auc", "std_train_cv_auc"]])
results_df.to_csv(...)
```
This turns panel-size results into a DataFrame, displays performance, and saves it.

### Lines 183-187
```python
best_auc = results_df["mean_train_cv_auc"].max()
eligible = results_df[results_df["mean_train_cv_auc"] >= best_auc - 0.01]
chosen = eligible.sort_values(["panel_size", "mean_train_cv_auc"], ascending=[True, False]).iloc[0]
FINAL_FEATURES = chosen["features"]
```
This chooses the smallest panel within 0.01 AUC of the best panel. The idea is parsimony: if a smaller panel performs almost as well as a larger one, prefer the smaller panel.

### Lines 189-194
Print the locked final panel size, training CV AUC, and features.

### Lines 200-211
```python
final_model = Pipeline([... StandardScaler(), LogisticRegression(penalty="l2") ...])
final_model.fit(X_train[FINAL_FEATURES], y_train)
```
This defines and fits the final model using the selected features and the training set.

### Lines 213-219
```python
test_probs = final_model.predict_proba(X_test[FINAL_FEATURES])[:, 1]
test_preds = (test_probs >= 0.5).astype(int)
test_auc = roc_auc_score(y_test, test_probs)
test_acc = accuracy_score(y_test, test_preds)
test_bal_acc = balanced_accuracy_score(y_test, test_preds)
cm = confusion_matrix(y_test, test_preds)
```
This evaluates the model on the frozen held-out test set.

- `predict_proba(... )[:, 1]` extracts probability/score for class 1 = ME/CFS.
- `test_preds` converts scores into hard labels using threshold 0.5.
- AUC evaluates ranking.
- Accuracy evaluates thresholded correctness.
- Balanced accuracy balances sensitivity and specificity.
- Confusion matrix counts TN/FP/FN/TP.

### Lines 221-228
Print the held-out test metrics and classification report.

### Lines 234-243
```python
coef = pd.Series(final_model.named_steps["logreg"].coef_[0], index=FINAL_FEATURES).sort_values(key=abs, ascending=False)
intercept = final_model.named_steps["logreg"].intercept_[0]
```
This extracts the logistic regression coefficients and intercept. Sorting by absolute coefficient shows which features have the largest modeled effects.

### Lines 245-264
These save the final panel, coefficients, and summary CSV files.

---

## Cell 2: Select best tradeoff panel

### Lines 5-15
Copies `results_df`, sorts by panel size, and calculates performance gains from adding features.

- `.diff()` compares each panel to the previous size.
- `auc_gain_per_added_feature` tells how much AUC improvement each added metabolite buys.

This is useful for identifying diminishing returns.

### Lines 18-28
Defines the best AUC and selects eligible panels that meet two criteria:

1. At least 95% of best AUC.
2. Mean CV AUC at least 0.85.

This rule favors a panel that is accurate enough but not unnecessarily large.

### Lines 30-39
If no panel meets those criteria, it chooses the highest-AUC panel. If some panels qualify, it chooses the smallest qualifying panel.

### Lines 41-65
Stores the selected tradeoff panel as `TRADEOFF_FEATURES`, prints the tradeoff table, and prints the selected features.

Your notebook selected a 10-metabolite panel.

---

## Cell 3: Held-out test evaluation of the 10-metabolite tradeoff panel

### Lines 5-14
Reimports model and metric tools. This is redundant because Cell 1 already imported most of them, but harmless.

### Lines 16-25
Creates `tradeoff_model`: StandardScaler followed by L2 logistic regression.

### Line 27
Fits the model on training subjects using only `TRADEOFF_FEATURES`.

### Lines 29-30
Predicts held-out ME/CFS scores and turns them into labels using threshold 0.5.

### Lines 32-41
Prints test AUC, accuracy, balanced accuracy, confusion matrix, and classification report.

Your output:

```text
AUC: 0.8588
Accuracy: 0.7907
Balanced accuracy: 0.7569
Confusion matrix:
[[10  6]
 [ 3 24]]
```

Interpreting the confusion matrix:

- 10 controls correctly called controls.
- 6 controls incorrectly called ME/CFS.
- 3 ME/CFS incorrectly called controls.
- 24 ME/CFS correctly called ME/CFS.

So at threshold 0.5, sensitivity is strong but specificity is more modest.

### Lines 43-52
Extracts and displays coefficients/intercept for the 10-panel model.

---

## Cell 4: Save locked 10-panel and repeated-split robustness

### Lines 5-18
Imports packages, defines `LOCKED_PANEL = TRADEOFF_FEATURES`, sets output directory, and creates it.

### Lines 20-28
Saves the locked panel feature list and coefficient file.

### Lines 30-54
Runs 100 repeated stratified 80/20 splits. For each split:

1. Make a new train/test split.
2. Fit scaler + L2 logistic regression on training set.
3. Predict probabilities on test set.
4. Save test AUC.

This asks: if the exact train/test split changes, does the locked panel still perform reasonably?

### Lines 56-65
Creates a summary DataFrame with mean, median, standard deviation, min, max, and the original seed-42 held-out AUC.

### Lines 67-72
Displays and saves robustness summary.

### Lines 74-85
Plots histogram of AUC values across repeated splits. The vertical lines show mean and median.

### Lines 87-91
Prints summary values.

Important interpretation: repeated-split AUC is not the same as true external validation, but it is a useful internal robustness check.

---

## Cell 5: Permutation test for 10-metabolite panel

### Lines 1-7
Imports tools.

### Lines 8-16
Defines the real model: scaler + L2 logistic regression.

### Lines 18-24
Computes real 5-fold CV AUC on the real labels.

### Lines 26-40
Repeats 1000 times:

1. Randomly shuffle `y`.
2. Cross-validate the same model against shuffled labels.
3. Store the resulting AUC.

### Line 44
```python
p_value = np.mean(perm_aucs >= real_auc)
```
This is the fraction of random-label models that scored at least as well as the real-label model.

Your output showed:

```text
Real AUC: 0.8531
Permutation p-value: 0.0
```

That means none of the 1000 shuffled-label runs matched the real AUC. More formally, you should report this as `p < 0.001`, not exactly zero.

### Lines 46-51
Prints the real AUC/p-value and plots the null distribution.

---

## Cell 6: Bootstrap confidence interval for held-out AUC

### Lines 3-6
Creates an empty list for bootstrapped AUCs and a reproducible random number generator.

### Lines 7-23
Repeats 2000 times:

1. Sample test-set indices with replacement.
2. Skip samples containing only one class because AUC requires both classes.
3. Compute AUC for that resampled test set.
4. Store it.

### Lines 25-29
Calculates 2.5th and 97.5th percentiles as a 95% CI.

Your output:

```text
Test AUC: 0.8588
95% CI: 0.7285 - 0.9538
```

This is wide because the test set has only 43 people.

---

## Cell 7: Mann-Whitney p-values for locked metabolites

### Lines 1-3
Imports `mannwhitneyu` and creates an empty result list.

### Lines 5-15
For each locked metabolite:

- `case`: values in ME/CFS subjects.
- `ctrl`: values in controls.
- `mannwhitneyu(case, ctrl)`: nonparametric group comparison.
- Store metabolite name and p-value.

### Line 17
Displays metabolites sorted by p-value.

This is not the model itself. It is a univariate sanity check: do individual metabolites differ between groups?

---

## Cell 8: Mann-Whitney + Cohen's d + FDR

### Lines 1-3
Imports Mann-Whitney, multiple-testing correction, and numpy.

### Lines 5-17
Defines `cohens_d(x, y)`.

Mathematically:

```text
pooled_sd = sqrt(((nx-1)*var_x + (ny-1)*var_y) / (nx + ny - 2))
d = (mean_x - mean_y) / pooled_sd
```

Here, `x` is cases and `y` is controls, so positive `d` means higher in ME/CFS.

### Lines 19-32
Loops through each panel metabolite, computes Mann-Whitney p-value and Cohen's d.

### Lines 34-39
Creates a DataFrame and applies Benjamini-Hochberg FDR correction.

### Lines 41-43
Sorts by FDR and displays.

---

## Cell 9: Random forest feature importance

### Lines 1-9
Creates a random forest classifier with 1000 trees. `class_weight="balanced"` compensates for class imbalance. `n_jobs=-1` uses all CPU cores.

### Line 11
Fits the forest on the training set using all metabolites.

### Lines 13-18
Creates a table of feature importances and displays the top 30.

Random forest importance is not the same as logistic regression coefficient. It measures how much each feature helps tree splits across the forest.

---

## Cell 10: Overlap between RF top 20 and locked panel

### Lines 1-6
Creates a set of top 20 random forest metabolites and checks which locked-panel metabolites are in it.

### Lines 8-12
Prints overlap count and feature names.

Your output showed 6/10 overlap, which supports that many locked metabolites are also important under another method.

---

## Cell 11: Mutual information feature ranking

### Lines 1-7
Computes mutual information between each metabolite and disease label.

Mutual information measures dependence, including nonlinear dependence. Higher MI means the feature contains more information about the label.

### Lines 9-14
Creates and displays a ranked MI table.

---

## Cell 12: Overlap between MI top 20 and locked panel

Same structure as Cell 10, but using mutual information instead of random forest. Your output showed only 1/10 overlap. This means MI ranking disagrees with the LASSO/consensus panel more than random forest does.

---

## Cell 13: Test RF top-10 features with logistic regression

### Lines 1-5
Extracts and prints the random forest top 10 metabolites.

### Lines 7-15
Creates a scaler + L2 logistic regression model.

### Lines 17-22
Fits that model using RF top-10 features and reports test AUC.

This is a comparison: if you chose features by random forest instead of LASSO stability, how good would the held-out AUC be?

---

## Cell 14: Test MI top-10 features with logistic regression

Same as Cell 13, but using mutual-information top 10 features.

---

## Cell 15: SHAP-style contribution scores for logistic regression

### Lines 1-5
The comments explain the idea: for a linear/logistic model, each feature's contribution to log-odds is:

```text
standardized_feature_value × coefficient
```

### Lines 10-13
```python
X_scaled = tradeoff_model.named_steps["scaler"].transform(X[TRADEOFF_FEATURES])
```
This uses the trained scaler from `tradeoff_model` to standardize all subjects' locked-panel values.

### Line 15
Extracts logistic regression coefficients.

### Line 18
```python
contrib = X_scaled * coefs
```
This computes each subject's feature-level contribution to the model log-odds.

If a subject has high standardized value for a metabolite with positive coefficient, the contribution is positive. If the coefficient is negative, high values reduce ME/CFS log-odds.

### Lines 20-23
Turns the contribution matrix into a DataFrame.

### Lines 25-31
Computes mean absolute contribution across subjects. This is a global importance score.

### Lines 33-40
Displays and plots global importance.

Important: this is “SHAP-like,” not actual SHAP. For linear models, it is a reasonable coefficient-times-value decomposition, but it does not include all SHAP baseline conventions.

---

## Cell 16: Shape, class counts, means, standard deviations

### Lines 1-4
Prints panel data shape and class counts.

### Lines 6-18
Displays means and standard deviations for each locked metabolite. These constants matter because logistic regression uses standardized values.

If you want to manually compute the score for a subject, you need:

1. Raw metabolite value.
2. Training-set mean.
3. Training-set standard deviation/scale.
4. Logistic coefficient.
5. Intercept.

---

## Cell 17: Logistic regression parameter dump

### Line 1
```python
tradeoff_model.named_steps["logreg"].get_params()
```
Displays the hyperparameters of the logistic regression model. Useful for reproducibility.

---

## Cell 18: NB1 final export

### Lines 5-12
Creates a final export folder called `MECFS_FINAL_SIGNATURES/NB1_LOCKED_10_PANEL`.

### Lines 18-23
Saves the locked 10-feature panel.

### Lines 29-34
Saves the 10-metabolite coefficients.

### Lines 40-45
Saves the model intercept.

### Lines 51-58
Attempts to save scaler constants:

```python
"mean": scaler.mean_,
"scale": scaler.scale_
```

Important bug: `scaler` is not defined earlier as a standalone variable in this notebook. The scaler lives inside `tradeoff_model`. The safe fix is:

```python
scaler_10 = tradeoff_model.named_steps["scaler"]

pd.DataFrame({
    "feature": TRADEOFF_FEATURES,
    "mean": scaler_10.mean_,
    "scale": scaler_10.scale_
})
```

### Lines 64-67
Saves repeated-split robustness summary.

### Lines 73-80
Saves permutation test results.

### Lines 83-84
Prints export folder.

---

## Cell 19: Reusable bootstrap AUC CI function

This repeats Cell 6 in a cleaner function form.

### Lines 3-4
Sets `probs = test_probs` and defines `bootstrap_auc_ci`.

### Lines 5-8
Creates RNG and converts inputs to numpy arrays.

### Lines 10-20
Resamples test-set indices with replacement, skips invalid one-class samples, and stores AUC.

### Lines 22-25
Returns mean, lower 2.5th percentile, and upper 97.5th percentile.

### Lines 27-31
Calls the function and prints formatted results.

---

## Cell 20: ROC and precision-recall curves

### Lines 3-8
Imports ROC and PR curve functions.

### Lines 12-22
Computes and plots ROC curve.

- `fpr`: false positive rate.
- `tpr`: true positive rate / sensitivity.
- `roc_thresholds`: thresholds used for ROC.
- diagonal dashed line: random classifier.

### Lines 26-41
Computes and plots precision-recall curve.

- Precision = PPV.
- Recall = sensitivity.
- Average precision summarizes PR performance.

PR curves are especially helpful when one class is rarer than the other.

---

## Cell 21: Youden's J threshold and threshold metrics

### Lines 5-8
Computes `j_scores = tpr - fpr`, finds the maximum, and gets the corresponding threshold.

### Line 10
Classifies subjects using the optimal threshold rather than 0.5.

### Lines 12-15
Unpacks confusion matrix into TN, FP, FN, TP.

### Lines 17-27
Computes sensitivity, specificity, PPV, NPV, accuracy, and balanced accuracy.

Your output:

```text
Optimal threshold: 0.581
Sensitivity: 0.889
Specificity: 0.688
PPV: 0.828
NPV: 0.786
Accuracy: 0.814
Balanced accuracy: 0.788
TN=11, FP=5, FN=3, TP=24
```

The threshold increases sensitivity while still keeping specificity moderate.

---

## Cell 22: Score distribution summary on test set

### Lines 4-7
Creates a DataFrame with true labels and predicted ME/CFS probabilities.

### Lines 9-13
Prints descriptive statistics separately for controls and ME/CFS.

### Lines 15-22
Prints highest control score, lowest ME/CFS score, and optimal threshold.

This directly shows overlap between groups. In your result:

- Highest control score ≈ 0.884.
- Lowest ME/CFS score ≈ 0.157.

So the groups overlap; this is not a perfect clinical diagnostic test.

---

## Cell 23: Inconclusive range search

### Lines 4-9
Creates a score DataFrame and possible threshold grid from 0.10 to 0.95.

### Lines 11-37
For every pair of `low_cut` and `high_cut`:

- Scores below `low_cut` are called low/control-like.
- Scores above `high_cut` are called high/ME-CFS-like.
- Scores in between are inconclusive.
- It calculates how many people are confidently called and how often those calls are correct.

### Lines 39-44
Sorts threshold pairs by high-confidence correctness and coverage.

This is exploring a triage-style test:

```text
low score → likely control / not ME-CFS-like
middle score → inconclusive
high score → likely ME-CFS-like
```

The tradeoff is that stricter thresholds give better confidence but classify fewer people.

---

# Notebook 2: `2_FINAL_metabolite_interpretation.ipynb`

## Cell 1: Markdown overview

The markdown explicitly says Notebook 2 should not fit/export new coefficients for the locked 10-metabolite model. That is important. Notebook 1 owns the locked 10-metabolite coefficients/intercept. Notebook 2 is for interpretation, downstream analyses, and multi-omics exploration.

---

## Cell 2: Imports and paths

### Lines 5-12
Imports basic utilities and suppresses warnings.

- `itertools`: used for modality combinations.
- `Path`: safer path handling.
- `Counter`: counts feature selections.

### Lines 14-19
Imports numpy, pandas, plotting, Mann-Whitney, and `expit`. `expit` is the sigmoid function:

```text
expit(x) = 1 / (1 + exp(-x))
```

It is imported but not heavily used in the visible code.

### Lines 21-33
Imports scikit-learn tools for CV, pipelines, column preprocessing, scaling, imputation, feature selection, logistic regression, and AUC.

### Lines 35-47
Defines file paths for metadata, metabolomics, Quest labs, immune data, KEGG, and species abundance.

### Lines 48-54
Creates output folders and sets random seed.

### Lines 56-58
Prints current folder and whether key files exist.

---

## Cell 3: Define locked 10-metabolite panel

### Lines 7-18
Hard-codes the 10 metabolites from Notebook 1. This is allowed because the panel list is fixed.

### Line 20
Returns the length, confirming there are 10 features.

Important: this cell only defines the feature names. It does not recompute the 10-metabolite coefficients.

---

## Cell 4: Load baseline metabolomics and labels

This repeats the Notebook 1 loading logic.

### Lines 5-14
Loads metadata/metabolomics and transposes metabolomics so rows become samples and columns become metabolites.

### Lines 16-26
Extracts participant IDs and timepoints, then keeps timepoint 1.

### Lines 28-35
Creates metadata participant IDs, keeps baseline metadata, and merges metadata with metabolomics.

### Lines 37-41
Creates label vector `y`, mapping ME/CFS to 1 and Control to 0.

### Lines 43-46
Builds `X_panel` using only the 10 locked metabolites and converts values to numeric.

### Lines 48-50
Prints data shape, class counts, and missing values. Your output showed `(214, 10)`, 135 ME/CFS, 79 controls, and 0 missing.

---

## Cell 5: Load Notebook 1 score file

### Lines 7-12
Defines possible paths where the Notebook 1 score file might live.

### Line 14
Finds the first existing score file.

### Lines 16-20
Raises an error if no score file exists. This enforces that Notebook 2 should load Notebook 1 scores, not refit them.

### Line 22
Reads the score CSV.

### Lines 25-26
If the score column is named `score`, rename it to `mecfs_score`.

### Lines 28-31
Checks that required columns exist: `participant_id` and `mecfs_score`.

### Lines 33-38
If the score file lacks `label`, merge labels from the current dataset.

### Lines 40-42
Prints loaded path, shape, and first few rows.

Your output showed a 214-row score table.

---

## Cell 6: Plot 10-metabolite score distribution

### Lines 5-19
Plots two histograms:

- Controls' `mecfs_score`.
- ME/CFS subjects' `mecfs_score`.

### Lines 21-27
Labels the plot, saves it, and displays it.

This visualizes how separated the two score distributions are.

---

## Cell 7: Merge score with metadata

### Lines 5-10
Merges Notebook 1 score table with raw metadata by participant/sample ID.

### Lines 12-13
Prints merged shape and displays rows.

This lets you ask: does the score correlate with age, BMI, diet variables, etc.?

---

## Cell 8: Correlate numeric metadata with metabolite score

### Line 5
Finds numeric metadata columns.

### Lines 8-19
For every numeric metadata column except `label` and `mecfs_score`:

1. Drop rows with missing score/metadata values.
2. Require more than 20 observations.
3. Compute Spearman correlation with score.

Spearman correlation measures monotonic association using ranks, so it is more robust than Pearson if distributions are non-normal.

### Lines 21-26
Creates a table sorted by absolute correlation and displays the top 30.

Your output showed weak correlations, e.g. age ~0.143 and BMI ~0.112, suggesting the score is not just age/BMI.

---

## Cell 9: Metadata-only model

### Lines 5-11
Uses age, BMI, and gender if present. Splits them into numeric and categorical sets.

### Lines 13-23
Creates a `ColumnTransformer`:

- Numeric columns: median imputation + standard scaling.
- Categorical columns: most-frequent imputation + one-hot encoding.

One-hot encoding turns categories into binary indicator columns. `drop="first"` avoids redundant columns.

### Lines 25-28
Creates metadata-only logistic regression model.

### Lines 30-31
Runs 5-fold CV AUC.

### Lines 33-37
Prints fold AUCs, mean, std, and missing counts.

Your metadata-only mean AUC was ~0.588, so demographics alone are weak predictors.

---

## Cell 10: Combined 10 metabolites + metadata model

### Lines 6-10
Combines `X_panel` with age/BMI/gender. Defines numeric and categorical columns.

### Lines 12-22
Builds preprocessing pipeline like Cell 9.

### Lines 24-32
Defines combined logistic regression model.

### Lines 34-44
Cross-validates the combined model. Your mean CV AUC was ~0.836.

Interpretation: adding simple metadata did not dramatically outperform the metabolite panel, and metadata alone was weak.

---

## Cell 11: Helper functions for multi-omics loading

### Lines 5-6
```python
def strip_tp(x):
    return re.sub(r"_tp\d+$", "", str(x))
```
Removes timepoint suffix from sample IDs.

### Lines 8-10
```python
def get_tp(x):
    m = re.search(r"_tp(\d+)$", str(x))
    return m.group(1) if m else np.nan
```
Extracts the timepoint number if present; otherwise returns missing.

### Lines 12-36
`load_transposed_omics(path, prefix, min_complete=0.80)` loads an omics matrix in the same transposed style as metabolomics.

Step-by-step:

- Read CSV.
- Treat first column as feature-name column.
- Transpose so samples are rows.
- Parse participant ID and timepoint.
- If timepoints exist, keep timepoint 1.
- Drop sample/timepoint columns.
- Set participant ID as index.
- Convert values to numeric.
- Keep features with more than 80% non-missing values.
- Prefix column names with modality, e.g. `metab__`, `quest__`, `species__`.

The prefix is very important because different modalities might have overlapping feature names.

---

## Cell 12: Build multi-omics modalities

### Lines 5-9
Builds `label_df`, indexed by participant ID, with disease label `y`.

### Line 11
Creates an empty dictionary called `modalities`.

### Lines 14-16
Loads all metabolomics, prefixes names with `metab__`, keeps only the locked 10 metabolite columns, and stores them as `locked_metab10`.

### Lines 19-21
Builds metadata covariates: age, BMI, gender. Gender is one-hot encoded with `pd.get_dummies`.

### Lines 24-39
Conditionally loads Quest, immune percentage, immune residue, KEGG, and species abundance if files exist.

### Lines 41-43
Prints each loaded modality and shape.

Your output:

```text
locked_metab10 (215, 10)
metadata (249, 3)
quest (242, 37)
immune_percentage (239, 311)
immune_residue (239, 311)
kegg (238, 4429)
species (238, 118)
```

---

## Cell 13: Test modality combinations with internal feature selection

### Lines 5-9
Loops over combinations of 1 to 4 modalities.

### Lines 11-15
For each combination, concatenates feature matrices, joins labels, and separates `Xc` and `yc`.

### Lines 17-18
Skips combinations with too few subjects or only one class.

### Line 20
Uses `k = min(50, number of features)`, so at most 50 features are selected.

### Lines 22-33
Builds a pipeline:

1. Median imputation.
2. Standard scaling.
3. `SelectKBest(f_classif, k=k)`.
4. L2 logistic regression.

`f_classif` is an ANOVA F-test. It scores each feature by how strongly the class means differ relative to within-class variance.

### Lines 35-42
Cross-validates the entire pipeline. Since `SelectKBest` is inside the pipeline, feature selection is done separately inside each training fold. That avoids CV leakage for this specific combination-screening step.

### Lines 44-51
Stores combination name, subject count, feature count, selected-k, mean AUC, and std AUC.

### Lines 53-60
Sorts combinations by mean AUC and saves results.

---

## Cell 14: Build one multi-omics matrix

### Lines 6-17
Concatenates selected modalities into `all_modalities`:

- locked 10 metabolites
- Quest
- immune percentage
- immune residue
- species
- metadata

KEGG is excluded here, probably because it has thousands of features and may be too high-dimensional/noisy.

### Lines 19-22
Joins labels and separates `X_multi` and `y_multi`.

### Lines 24-25
Prints shape and class counts. Your output: `X_multi: (249, 790)`, labels `{1: 153, 0: 96}`.

---

## Cell 15: Multi-omics stability selection with LASSO

### Lines 5-8
Sets LASSO `C_GRID`, repeats, and creates a feature counter.

### Lines 10-12
Defines `clean_feature_name`, but it just returns the input feature unchanged. It is effectively unused.

### Lines 14-37
For 20 repeats of 5-fold CV and 5 C-values:

1. Take training indices.
2. Fit imputer + scaler + L1 logistic regression.
3. Identify nonzero coefficients.
4. Update selection counts.

Total selection opportunities: `20 × 5 × 5 = 500`.

### Lines 39-48
Builds feature stability table, computes frequency, sorts, displays, and saves.

Important caution: this stability selection uses all `X_multi`/`y_multi` to choose the panel. Later CV of panel sizes uses the resulting fixed panel. That can be optimistic because the panel was chosen using the full data. It is okay for exploratory panel discovery, but for a strict validation claim you would need nested CV or an external dataset.

---

## Cell 16: Evaluate multi-omics panel sizes

### Lines 5-9
Tests panel sizes 5, 10, 15, 20, and 30 using the top features from `multi_stability`.

### Lines 11-20
Defines imputer + scaler + L2 logistic regression model.

### Lines 22-28
Cross-validates each panel size.

### Lines 30-39
Stores results and saves them.

Your output showed:

```text
5 features:  AUC ~0.826
10 features: AUC ~0.910
15 features: AUC ~0.917
20 features: AUC ~0.926
30 features: AUC ~0.929
```

Interpretation: performance increases with larger panels, but 15 is a reasonable interpretability/performance tradeoff. However, because feature selection was done globally first, these CV numbers are exploratory/optimistic.

---

## Cell 17: Lock final multi-omics panel

### Lines 6-7
Sets final multi-omics panel size to 15 and takes the top 15 stability-selected features.

### Lines 9-11
Prints the locked 15 features.

Your selected panel includes metabolomics, Quest Vitamin B12, immune NK cell features, and several species/microbiome features.

---

## Cell 18: Final multi-omics score model and coefficients

### Lines 7-16
Defines final multi-omics pipeline: imputer, scaler, L2 logistic regression.

### Line 18
Fits the final model on all `X_multi` subjects using the 15-feature panel.

### Line 20
Predicts probabilities for the same subjects used to train the model.

### Lines 22-26
Creates subject-level multi-omics score table.

### Lines 28-32
Extracts coefficients and sorts by absolute value.

### Lines 34-36
Prints apparent full-data AUC and displays scores/coefficients.

Important: `roc_auc_score(y_multi, multi_probs)` here is an apparent/training AUC, not a held-out validation AUC. The notebook labels it correctly as “Apparent full-data.” Do not present that number as final validation.

---

## Cell 19: Multi-omics score distribution

### Lines 5-26
Plots control and ME/CFS multi-omics scores as histograms.

### Lines 28-32
Runs Mann-Whitney test comparing multi-omics score distributions.

Your p-value was extremely small, meaning the score distributions are very different in-sample.

---

## Cell 20: Compare 10-metabolite score vs multi-omics score

### Lines 5-12
Merges Notebook 1 score and multi-omics score by participant.

### Line 14
Computes `delta = multiomics_score - mecfs_score`.

### Line 16
Computes correlation between the two scores.

### Lines 17-19
Prints and displays results.

Your score correlation was ~0.821, meaning the two scores are strongly related but not identical.

---

## Cell 21: Score correlation plots

### Lines 5-11
Scatter plot of 10-metabolite score vs 15-feature multi-omics score.

### Lines 13-22
Same scatter plot colored/labeled by class.

This visually shows whether multi-omics mostly agrees with metabolomics, and where subjects shift.

---

## Cell 22: Repeated train/test robustness checks

### Lines 5-11
Sets `N_SPLITS = 500` and defines two models to compare:

1. 10-metabolite predictors only.
2. 15-feature multi-omics panel.

### Lines 12-37
For each model and each seed:

1. Make stratified 80/20 split.
2. Fit imputer + scaler + L2 logistic regression.
3. Predict test probabilities.
4. Store AUC.

### Lines 38-51
Summarizes the distribution of AUCs with mean, median, std, min/max, and quantiles.

### Lines 53-61
Prints and displays robustness summary.

Your output:

```text
10-metabolite mean AUC ≈ 0.859, 5th-95th ≈ 0.757-0.942
15-multiomics mean AUC ≈ 0.910, 5th-95th ≈ 0.847-0.968
```

Interpretation: the multi-omics panel appears stronger internally, but this is still not external validation and may inherit selection optimism.

---

## Cell 23: Permutation test for locked multi-omics panel

### Lines 5-13
Creates one stratified 80/20 split for the multi-omics panel.

### Lines 15-27
Fits real model and computes real held-out AUC.

### Lines 29-47
Repeats 1000 times:

1. Shuffle training labels only.
2. Fit model on shuffled labels.
3. Predict the same real test set.
4. Store AUC.

### Lines 49-55
Computes permutation p-value with `+1` smoothing and prints real/null stats.

Your output:

```text
Real AUC: 0.9202
Null mean AUC: 0.5046
Permutation p-value: 0.000999
```

This supports that the selected panel has real label signal under this internal split.

### Lines 57-65
Plots null AUC distribution with real AUC as a vertical line.

---

## Cell 24: No-gut sensitivity analysis

### Line 5
Creates `NO_GUT_PANEL` by removing features starting with `species__`.

### Lines 7-13
Prints original and no-gut sizes and lists removed species features.

### Lines 15-39
Runs 500 repeated splits using only non-species features.

### Lines 41-48
Summarizes AUC distribution.

Your no-gut mean AUC was ~0.864, similar to the 10-metabolite model and lower than the full multi-omics panel. This suggests gut/species features contribute to the internal multi-omics boost.

---

## Cell 25: Final exports

### Lines 7-11
Saves locked 10-metabolite panel list.

### Lines 13-17
Saves locked 15-feature multi-omics panel list and modality labels.

### Lines 19-21
Saves subject scores, multi-omics stability table, and panel size CV results.

### Lines 23-26
Saves multi-omics coefficients.

### Line 28
Saves robustness summary.

### Lines 30-32
Saves score correlation summary.

### Lines 34-39
Prints export folder and file list.

---

## Cell 26: 10-panel shape, class counts, means, standard deviations

Same purpose as Notebook 1 Cell 16. It confirms the 10-metabolite matrix shape and feature distribution.

---

## Cell 27: NB2 final export

### Lines 5-12
Creates export folder `MECFS_FINAL_SIGNATURES/NB2_MULTIOMICS_INTERPRETATION`.

### Lines 18-23
Saves locked 15-feature multi-omics panel.

### Lines 29-34
Saves multi-omics coefficients.

### Lines 40-47
Saves multi-omics intercept.

### Lines 53-56
Saves subject scores.

### Lines 62-65
Saves multi-omics feature stability.

### Lines 71-74
Saves panel-size CV results.

### Lines 80-83
Saves robustness summary.

### Lines 89-97
Saves permutation test summary.

### Lines 103-107
Saves score correlation.

### Lines 110-111
Prints export folder.

---

## Cell 28: Fresh held-out split for chosen panel

### Lines 1-6
Imports tools.

### Line 8
Sets `PANEL = MULTIOMICS_PANEL`. The comment says you could switch to `LOCKED_PANEL`, but as written it uses multi-omics.

### Lines 10-11
Defines model matrix and labels.

### Lines 13-19
Creates stratified 80/20 split.

### Lines 21-30
Defines imputer + scaler + L2 logistic regression pipeline.

### Line 32
Fits on training split.

### Line 34
Gets probability for class 1. This version is extra-safe because it finds the index of class `1` in `pipe.classes_`.

### Lines 36-38
Prints classes, test size, and AUC.

Your output: test AUC ≈ 0.920 on n=50.

---

## Cell 29: Bootstrap CI for the multi-omics held-out AUC

Same function as Notebook 1 Cell 19, now applied to the multi-omics test split.

Your output:

```text
AUC: 0.920
95% CI: [0.836, 0.977]
```

---

## Cell 30: ROC and PR curves for multi-omics held-out split

Same logic as Notebook 1 Cell 20.

---

## Cell 31: Youden threshold metrics for multi-omics held-out split

Same logic as Notebook 1 Cell 21.

Your output:

```text
Optimal threshold: 0.848
Sensitivity: 0.710
Specificity: 1.000
PPV: 1.000
NPV: 0.679
Accuracy: 0.820
Balanced accuracy: 0.855
TN=19, FP=0, FN=9, TP=22
```

Important interpretation: 100% specificity and PPV happened on only 19 controls in this split. That is encouraging but fragile. It should not be claimed as clinical proof.

---

## Cell 32: Test-set score distribution summary for multi-omics

Same logic as Notebook 1 Cell 22.

Your output showed:

- Highest control score ≈ 0.827.
- Lowest ME/CFS score ≈ 0.336.
- Optimal threshold ≈ 0.848.

That means the threshold was just above the highest control score, which explains why specificity was 1.000 on that split.

---

## Cell 33: Inconclusive range search for multi-omics

Same logic as Notebook 1 Cell 23. It searches low/high thresholds that create:

- low-confidence-control call,
- high-confidence-ME/CFS call,
- middle inconclusive zone.

This is useful for thinking about a future triage-style score, but with this sample size, the thresholds should be treated as exploratory.

---

# Important issues / fixes / interpretation cautions

## 1. Notebook 1 Cell 18 has a likely bug
This code:

```python
"mean": scaler.mean_,
"scale": scaler.scale_
```

should probably be:

```python
scaler_10 = tradeoff_model.named_steps["scaler"]

pd.DataFrame({
    "feature": TRADEOFF_FEATURES,
    "mean": scaler_10.mean_,
    "scale": scaler_10.scale_
})
```

Otherwise the export depends on whether some old variable named `scaler` exists in the notebook memory.

## 2. Notebook 2 multi-omics panel performance is promising but optimistic
The multi-omics panel was selected using all available data, then evaluated internally. That means the AUCs are useful for exploration, but for strict publication language, call them internal/optimistic unless you run nested CV or external validation.

## 3. The 10-metabolite panel is cleaner methodologically
Notebook 1 does selection on training data only, then evaluates once on a held-out test split. That makes the 10-metabolite analysis easier to defend.

## 4. 100% specificity/PPV in Notebook 2 should be handled carefully
The multi-omics threshold gave zero false positives in one held-out split:

```text
TN=19, FP=0, FN=9, TP=22
```

But 19 controls is small. A single additional false positive would drop specificity from 1.000 to 0.950. So this is not overfit by itself, but it is a fragile estimate.

## 5. The inconclusive-zone idea is valid, but threshold estimates are unstable
The notebooks search low/high thresholds to improve confidence. That is conceptually reasonable. But the test set is small, so the exact cutoffs should be treated as exploratory.

## 6. Best publishable framing
The strongest framing is not:

> “We created a clinically deployable diagnostic test.”

A defensible framing is closer to:

> “We identified and internally validated a sparse metabolomic ME/CFS-associated signature with robust discrimination, and explored multi-omics extensions that improve internal performance and suggest biological axes for follow-up validation.”

---

# One-sentence summary of each notebook

Notebook 1 discovers and internally validates a locked 10-metabolite ME/CFS signature using training-only LASSO stability selection plus held-out logistic-regression evaluation.

Notebook 2 interprets that locked metabolite score, compares it against metadata and multi-omics extensions, and shows that a 15-feature multi-omics panel improves internal AUC but requires careful validation because feature selection is less strictly separated from evaluation.
