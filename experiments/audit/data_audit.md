# Data Audit Report

```json
{
  "splits": {
    "train_vs_val": {
      "exact_duplicates_in_second": 134,
      "near_duplicate_items_in_second": 138,
      "union_items": 138,
      "pct_of_second": 9.16
    },
    "train_vs_test": {
      "exact_duplicates_in_second": 152,
      "near_duplicate_items_in_second": 155,
      "union_items": 155,
      "pct_of_second": 10.29,
      "exact_dup_label_agreement": 0.967
    },
    "val_vs_test": {
      "exact_duplicates_in_second": 26,
      "near_duplicate_items_in_second": 27,
      "union_items": 27,
      "pct_of_second": 1.79
    },
    "train_internal_duplicates": 636,
    "train_label_conflicts": 7
  }
}
```