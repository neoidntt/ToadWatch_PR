# Dataset

Prepared data, one subfolder per set:

* `train/` - training data
* `validation/` - fitting / validation data
* `test/` - final testing data (locked, read-only)

| set | records | fraud | safe |
|---|---|---|---|
| train | 3287 | 1646 | 1641 |
| validation | 461 | 231 | 230 |
| test | 925 | 463 | 462 |

Rebuilt automatically whenever the data is prepared; open it from code with `toadwatch.dataset_access`.
The raw files you provided stay in the original four folders next to `ToadWatch_PR/`.
