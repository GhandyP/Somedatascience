# Name Gender Classifier

This project evaluates how well first names can predict gender association.
It is intended as a reproducible data-science portfolio project, not as a
claim about an individual's gender.

The planned data source is the SSA-derived `hadley/data-baby-names` dataset.
It covers the top 1,000 names per sex and year from 1880 through 2008, rather
than the complete SSA roster.

The data pipeline, model, and evaluation are being developed in later slices.
Results are pending, and the 1880-2008 coverage limitation will remain an
explicit part of the final analysis.

To regenerate the committed sample fixture, run `make regenerate-fixture` from
this directory. This is strictly offline: it reads only the existing
`data/raw/baby-names.csv` cache and fails if that file is absent. The selection
keeps the first 30 source rows per `(year, sex)` for 1880 and 1881, in global
source order. Output is UTF-8 CSV with the source header quoted, `name` and
`sex` fields quoted (with embedded quotes doubled), `year` and `percent` fields
unquoted, and LF (`\\n`) line endings.
