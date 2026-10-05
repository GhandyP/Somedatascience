# Hong Kong rainfall observations

This project ingests the Kaggle dataset [Hong Kong Weather Observation Summary 2010~2019](https://www.kaggle.com/datasets/act18l/hongkongrainfall), dataset identifier `act18l/hongkongrainfall`, version 2, uploaded by **Spiritmilk**. The Kaggle description attributes the observations to the Hong Kong Observatory; that upstream attribution has not been independently verified.

## Dataset terms and handling

The dataset is provided under the [Community Data License Agreement — Sharing, Version 1.0](https://cdla.dev/sharing-1-0/). Review the [license terms](https://cdla.dev/sharing-1-0/) before reuse or distribution. Attribute the dataset and license when sharing covered data; the project avoids publishing raw or row-level transformed observations. The repository's code license applies to code only, not this dataset.

The archive is pinned to Kaggle v2 and SHA-256 `73bba9f13dd13ba2c8b5e92f72b1da5b03496ceca326d157f3a0147274abea02`. Use `hk_rainfall.ingest.fetch_archive` to download it to a local `hk-rainfall/data/` cache; the archive is checked before atomic publication. `load_archive` verifies the same digest before opening the ZIP. Neither raw archive nor parsed rows belong in version control.

The parser decodes source CSV as GB18030, preserves the exact source headers in provenance (including `temparature` and `low visibility hour`), and validates the observed 17-column schema: `year`, `month`, and `day` supply the calendar components from which it derives the canonical analysis `date`. It checks non-empty input, unique increasing daily dates, continuous coverage, and the documented date endpoints. Downstream analysis should normalize source headers separately. In rainfall, `-` means not detected and is a non-event; `微量` denotes trace presence and is an event. Blank and `-` in other measurement columns are treated as missing measurements, not rainfall states.

## CLI setup, data, tests, and reproduction

From the repository root, enter the `hk-rainfall` project directory, then bootstrap the isolated environment and install the CLI and tests:

```sh
cd hk-rainfall
make setup
```

The setup target handles Python installations without `ensurepip`. If its pip bootstrap cannot use the host Python, create a seeded environment with `uv venv --seed .venv` and rerun `make setup`.

Fetch and verify the pinned archive explicitly (this is the only networked command):

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
