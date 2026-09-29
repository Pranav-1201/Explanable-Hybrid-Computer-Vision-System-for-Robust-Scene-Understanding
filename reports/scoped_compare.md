# Scoped (24 + other) vs served (67-class)

Scoped: T=0.6741, threshold 0.6407 (5th pct of correct validation confidence). Served threshold 0.4780. Same metric definitions for both (evaluation/field_metrics.py).

## test (n=1340, home photos=472)

| Metric | served 67-class | scoped 25-class |
|---|---|---|
| auto_tagged | 420 | 366 |
| right_tags | 377 | 334 |
| wrong_tags | 43 | 32 |
| tag_precision | 0.898 | 0.913 |
| home_tag_recall | 0.799 | 0.708 |
| non_home_false_tags | 23 | 17 |
| non_home_photos | 868 | 868 |
| reviewed | 920 | 974 |
| review_precision | 0.975 | 0.960 |

Tag precision 95% CI: served 0.865 to 0.923, scoped 0.879 to 0.937.

## field (n=72, home photos=26)

| Metric | served 67-class | scoped 25-class |
|---|---|---|
| auto_tagged | 24 | 20 |
| right_tags | 21 | 18 |
| wrong_tags | 3 | 2 |
| tag_precision | 0.875 | 0.900 |
| home_tag_recall | 0.808 | 0.692 |
| non_home_false_tags | 2 | 1 |
| non_home_photos | 46 | 46 |
| reviewed | 48 | 52 |
| review_precision | 0.979 | 0.942 |

Tag precision 95% CI: served 0.690 to 0.957, scoped 0.699 to 0.972.
