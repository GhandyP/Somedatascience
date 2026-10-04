# Hong Kong rainfall observations

This project ingests the Kaggle dataset [Hong Kong Weather Observation Summary 2010~2019](https://www.kaggle.com/datasets/act18l/hongkongrainfall), dataset identifier `act18l/hongkongrainfall`, version 2, uploaded by **Spiritmilk**. The Kaggle description attributes the observations to the Hong Kong Observatory; that upstream attribution has not been independently verified.

## Dataset terms and handling

The dataset is provided under the [Community Data License Agreement — Sharing, Version 1.0](https://cdla.dev/sharing-1-0/). Review the [license terms](https://cdla.dev/sharing-1-0/) before reuse or distribution. Attribute the dataset and license when sharing covered data; the project avoids publishing raw or row-level transformed observations. The repository's code license applies to code only, not this dataset.

The archive is pinned to Kaggle v2 and SHA-256 `73bba9f13dd13ba2c8b5e92f72b1da5b03496ceca326d157f3a0147274abea02`. Use `hk_rainfall.ingest.fetch_archive` to download it to a local `hk-rainfall/data/` cache; the archive is checked before atomic publication. `load_archive` verifies the same digest before opening the ZIP. Neither raw archive nor parsed rows belong in version control.

The parser decodes source CSV as GB18030, preserves the exact source headers in provenance (including `temparature` and `low visibility hour`), and validates the observed 17-column schema: `year`, `month`, and `day` supply the calendar components from which it derives the canonical analysis `date`. It checks non-empty input, unique increasing daily dates, continuous coverage, and the documented date endpoints. Downstream analysis should normalize source headers separately. In rainfall, `-` means not detected and is a non-event; `微量` denotes trace presence and is an event. Blank and `-` in other measurement columns are treated as missing measurements, not rainfall states.

## Setup and offline tests

From the repository root, create an isolated environment and install the pinned project and test dependencies:

```sh
python3 -m venv hk-rainfall/.venv
hk-rainfall/.venv/bin/python -m pip install -e 'hk-rainfall[test]'
```

If your Python installation omits `ensurepip`, create the same environment with `uv venv --seed hk-rainfall/.venv` instead.

Offline tests use synthetic data and mocked downloads; no Kaggle credentials or network access are needed:

```sh
cd hk-rainfall
.venv/bin/python -m pytest -q tests
```
