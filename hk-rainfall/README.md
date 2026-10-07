# Hong Kong rainfall observations

This project evaluates a next-day rainfall-event baseline from the Kaggle dataset [Hong Kong Weather Observation Summary 2010~2019](https://www.kaggle.com/datasets/act18l/hongkongrainfall), dataset identifier `act18l/hongkongrainfall`, version 2, uploaded by **Spiritmilk**. The Kaggle description attributes the observations to the Hong Kong Observatory; that upstream attribution has not been independently verified. There is no verified link to Kaggle's Playground Series S5E3 competition.

## Dataset terms and handling

The dataset is provided under the [Community Data License Agreement — Sharing, Version 1.0](https://cdla.dev/sharing-1-0/). Review the [license terms](https://cdla.dev/sharing-1-0/) before reuse or distribution. Attribute the dataset and license when sharing covered data; the project avoids publishing raw or row-level transformed observations. The repository's code license applies to code only, not this dataset.

The archive is pinned to Kaggle v2 and SHA-256 `73bba9f13dd13ba2c8b5e92f72b1da5b03496ceca326d157f3a0147274abea02` (85,061 bytes). Use `hk_rainfall.ingest.fetch_archive` to download it to a local `hk-rainfall/data/` cache; the archive is checked before atomic publication. `load_archive` verifies the same digest before opening the ZIP. Neither raw archive nor parsed rows belong in version control.

The parser decodes source CSV as GB18030, preserves the exact source headers in provenance (including `temparature` and `low visibility hour`), and validates the observed 17-column schema: `year`, `month`, and `day` supply the calendar components from which it derives the canonical analysis `date`. It checks non-empty input, unique increasing daily dates, continuous coverage, and the documented date endpoints. Downstream analysis should normalize source headers separately. In rainfall, `-` means not detected and is a non-event; `微量` denotes trace presence and is an event. Blank and `-` in other measurement columns are treated as missing measurements, not rainfall states.

## Next-day task and method

Each sample uses measurements and the rainfall event observed on day *t*, plus known cyclic calendar features for target day *t+1*, to predict whether rainfall occurs on day *t+1*. No weather measurement or rainfall observation from the target day is an input. Imputation and scaling are fitted on training data only, followed by logistic regression.

The split is chronological by **target date**:

| Split | Target-date range | Samples |
| --- | --- | ---: |
| Train | 2010-01-02–2016-12-31 | 2,556 |
| Validation | 2017-01-01–2018-12-31 | 730 |
| Test | 2019-01-01–2019-10-31 | 304 |

The source contains 3,591 daily observations, yielding 3,590 next-day samples. Comparisons use two simple baselines: a constant probability equal to training prevalence, and prior-day rainfall persistence. Precision, recall, F1, and accuracy classify probabilities at threshold 0.5. Average precision (AP), ROC-AUC, and Brier score evaluate ranking and probability quality. Higher AP, ROC-AUC, F1, and accuracy are better; lower Brier score is better.

## Aggregate results

Metrics below were reproduced from an explicitly fetched, checksum-verified local archive. Two reproductions wrote byte-identical aggregate JSON (SHA-256 `2c0f75579f3fa114c9f715c6ea63e47f153c0cf0f096aa489814a84024fa5b05`); the report was kept under `/tmp`, not added to the repository. All published results are aggregates only; no raw rows are published.

### Ranking and probability metrics

| Split | Model | AP | ROC-AUC | Brier (lower is better) |
| --- | --- | ---: | ---: | ---: |
| Validation | Logistic regression | 0.857 | 0.822 | 0.170 |
| Validation | Training-prevalence baseline | 0.593 | 0.500 | 0.242 |
| Validation | Prior-day persistence | 0.739 | 0.730 | 0.260 |
| Test | Logistic regression | 0.916 | 0.833 | 0.158 |
| Test | Training-prevalence baseline | 0.674 | 0.500 | 0.222 |
| Test | Prior-day persistence | 0.779 | 0.700 | 0.263 |

### Test metrics at probability threshold 0.5

| Model | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.775 | 0.907 | 0.836 | 0.760 |
| Training-prevalence baseline | 0.674 | 1.000 | 0.806 | 0.674 |
| Prior-day persistence | 0.805 | 0.805 | 0.805 | 0.737 |

This is an educational historical baseline, not an operational forecast. The reported test performance comes from one future test window and does not establish performance in other periods or operational conditions.

## Setup, data, tests, and reproduction

From the repository root, enter the `hk-rainfall` project directory, then bootstrap the isolated environment and install the CLI and tests:

```sh
cd hk-rainfall
make setup
```

The setup target handles Python installations without `ensurepip`. Dependency installation may access the Python package index; tests and local reproduction do not access the Kaggle network. If its pip bootstrap cannot use the host Python, create a seeded environment with `uv venv --seed .venv` and rerun `make setup`.

Fetch and verify the pinned Kaggle dataset explicitly (`make data` is the only command that fetches it):

```sh
make data
```

Run the offline synthetic tests:

```sh
make test
```

Reproduce the evaluation locally:

```sh
make reproduce
```

Reproduction requires an archive previously fetched into `data/`; it has no hidden network request. The default report is `reports/evaluation.json`. You can select paths directly with `python -m hk_rainfall.cli reproduce --archive PATH --output PATH` (or use `hk-rainfall` after installation). To fetch elsewhere, use `python -m hk_rainfall.cli fetch --archive PATH`.
