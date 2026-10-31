# Dataset v1

Prepared data, split 70 / 10 / 20, one subfolder per set:

* `train/` - training data
* `validation/` - fitting / validation data
* `test/` - final testing data (locked, read-only)

| set | records | fraud | safe |
|---|---|---|---|
| train | 2862 | 1432 | 1430 |
| validation | 410 | 204 | 206 |
| test | 818 | 409 | 409 |

Regenerated automatically by the data-preparation step; open it from code with `toadwatch.dataset_access`
(see `notebooks/v2_00_explore_dataset.ipynb`). The raw files you provided stay in the original four folders next to
`ToadWatch_PR/`; `dataset_access.raw_inventory()` lists them.
