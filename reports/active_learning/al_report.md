# Active Learning Report

All strategies use the same initial sample, pool, and untouched stratified test set.
Pool source labels are used only as a simulation oracle after a query selects a row.

| Strategy | Final accuracy | Final macro F1 | N reaching random final F1 | Saved labels |
|---|---:|---:|---:|---:|
| entropy | 0.6562 | 0.6517 | not reached | n/a |
| margin | 0.6562 | 0.6517 | not reached | n/a |
| random | 0.6667 | 0.6661 | 150 | 0 |
