# Name Gender Classifier

This project measures how much a first name, without a year, predicts the dominant gender associated with that name in US baby-name data. The result is better than simple baselines, but the difficult cases are exactly the names whose association changed over time. This is an analysis of historical name data, not a claim about any individual’s gender.

## The starting point

The project replaces a 95-line script that reported `accuracy 1.00`. That script fitted a `DictVectorizer` inside a helper and never returned it. A raw dictionary eventually reached `predict`, and the script crashed with `TypeError: float() argument must be a string or a real number, not 'dict'`.

The reported `1.00` was also calculated from ten hardcoded rows where the label was a deterministic function of the last letter. The metric was a property of the evaluation, not of the model.

## Data

The data is the SSA-derived US baby-names dataset mirrored by [`hadley/data-baby-names`](https://github.com/hadley/data-baby-names). It is a US Government work in the public domain under 17 U.S.C. 105. The pinned dataset contains 244,078 `(name, year)` rows over 6,782 names spanning 1880 through 2008. `make data` fetches it and verifies its pinned SHA-256:

```text
1259523fa76e5c18151a4c7612b854b22605d07127c596126940e2941ba15d3c
```

This is the top 1,000 names per sex per year, **not** the complete SSA roster. The series ends in 2008.

## Task and evaluation

There is one row per `(name, year)`. The label is the dominant gender for that year. The model receives the name and never the year. That mismatch is deliberate: it makes a gender-flipping name hard and measures whether a name-only model can generalise to names it has not seen during training.

The data is split by name, so no name appears on both sides. The test set contains 48,846 rows over 1,357 names.

| Method | Accuracy |
| --- | ---: |
| majority class | 0.4980 |
| last-letter rule | 0.8051 |
| name lookup | 0.4980 |
| **model** | **0.8461** |

The model beats the last-letter rule by **4.1 points**. That improvement is real and modest. The name-lookup baseline must degenerate to the majority baseline under a split by name: every test name is unseen. That degeneration is evidence that this task is largely a lookup problem when the same name is allowed on both sides of a split.

### Where it fails

| Ambiguity | Accuracy | Test rows |
| --- | ---: | ---: |
| nearly pure | 0.8539 | 47,093 |
| fairly skewed | 0.6258 | 318 |
| ambiguous | 0.6538 | 491 |
| very ambiguous | 0.6320 | 943 |
| genuinely 50/50 | too small to interpret | 1 |

The model is strong where a name has a clear gender association and falls toward chance where it does not. That is not a defect: a name that is genuinely 50/50 has no correct answer to predict. The ambiguous buckets contain hundreds of rows rather than thousands, so their accuracies are suggestive rather than precise. The 50/50 bucket contains one row and is reported as uninterpretable rather than as an accuracy.

### The `1.00`, reproduced

Fitting the honest pipeline on the legacy script’s own ten names and scoring those same ten names returns **exactly 1.0000**. The old number was reachable by anyone; it simply measured nothing. This reproduces the old result while showing why it is not evidence of generalisation.

### Leakage, measured

The table, target, and model are unchanged; only the split differs:

- split by name: **0.8461**
- split by row: **0.9669**
- inflation: **0.1208**

A row split lets the model see names in both training and test data. The name split is the relevant result for testing generalisation to unseen names.

## Why ambiguity exists

154 names hold a different dominant gender in different decades, including Whitney, Kelley, Presley, Jodie, Joan, Regan, Ricki, and Robbie. Several cross the full range from a female share of 0.0 to 1.0.

On the same split, names that changed gender are 4.71% of test rows but 12.09% of the model’s errors, an over-representation factor of 2.6. Accuracy on those names is 0.6053 over 2,303 rows, compared with 0.8580 over 46,543 rows for names that never moved.

The mechanism is direct: a name that moved carries different per-year labels, but the model sees only the name. No single name-only prediction can be right for every row of that name. The model is not failing arbitrarily; it fails where the task has no single correct answer.

## Reproduce

From a clean checkout:

```bash
make setup
make data
make test
make reproduce
```

The test suite makes no network calls. It needs the pinned dataset, which `make data` fetches once and verifies against its digest. Therefore a clean checkout cannot run the tests before that fetch. CI runs the same sequence on Python 3.11 and 3.13, and fails if `make reproduce` does not leave the committed reports byte-identical.

The committed evidence is [`reports/evaluation.txt`](reports/evaluation.txt), [`reports/drift.csv`](reports/drift.csv), and [`reports/drift-summary.txt`](reports/drift-summary.txt). All three are regenerated by `make reproduce`; CI checks that the reports are byte-identical, so the reported evaluation and drift numbers are verifiable rather than transcribed.

## Limitations

- The series ends in 2008.
- Coverage is limited to the top 1,000 names per sex per year.
- The label is a dominant-gender summary: a name that flipped is represented by one label per year, not by a distribution.
- The model captures orthographic association in US baby-name data. It makes no claim about any individual.
