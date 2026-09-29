# Review routing: margin and entropy (M2)

Question: does routing on top-1 minus top-2 margin, or on predictive entropy, catch
wrong tags that the calibrated confidence gate lets through?

Setup: served model, TTA on, temperature T = 0.5142. Thresholds are the 2nd/5th
percentile of margin (or 98th/95th of normalised entropy) among validation
predictions that are correct **and** already pass the confidence gate. The
validation split is the seed-42 10% of the train folder (n = 536), never trained
on. The test set (n = 1,340) and the 72 field photos are only scored.

"Right tag" = auto-tagged with the true home class. "Wrong tag" = any other
auto-tag, including any tag on a non-home photo.

## Tag precision and home recall

| Policy | Val tag prec. | Val recall | Test wrong / tagged | Test tag prec. | Test recall |
|---|---|---|---|---|---|
| confidence only (served) | 0.903 | 0.819 | 43 / 420 | 0.898 | 0.799 |
| + margin p1 (0.192) | 0.907 | 0.819 | 40 / 414 | 0.903 | 0.792 |
| + margin p2 (0.276) | 0.921 | 0.814 | 35 / 405 | 0.914 | 0.784 |
| + margin p5 (0.412) | 0.939 | 0.786 | 32 / 389 | 0.918 | 0.756 |
| + entropy p99 (0.436) | 0.907 | 0.814 | 39 / 409 | 0.905 | 0.784 |
| + entropy p95 (0.330) | 0.932 | 0.767 | 33 / 393 | 0.916 | 0.763 |

On the 72 field photos every row is identical: 24 auto-tagged, 3 wrong, precision
0.875, recall 0.808.

Reading it: each extra point of precision costs about as much recall. On test,
margin p2 removes 8 wrong tags and 15 tags in total (7 right ones). The 95%
interval on a precision near 0.90 with 420 tags is about +/-3 points, so the
differences between rows are inside the noise.

## Do better scores exist?

Misclassification AUROC (how well a score ranks right predictions above wrong
ones), single-centre-crop logits:

| Score | Val | Test | Field (n=68) |
|---|---|---|---|
| max softmax (served) | 0.866 | 0.857 | 0.748 |
| margin | 0.861 | 0.859 | 0.734 |
| negative entropy | 0.860 | 0.852 | 0.745 |
| max logit | 0.844 | 0.838 | 0.731 |
| energy | 0.821 | 0.818 | 0.702 |
| Mahalanobis (tied covariance, train embeddings) | 0.696 | 0.683 | 0.775 |

Nothing beats max softmax on the two large sets. Mahalanobis is best on the field
photos, but that is 68 photos with 14 wrong; it is worse on val and test.

## The two roadmap cases

| Photo | Prediction | Confidence | Margin | Normalised entropy |
|---|---|---|---|---|
| `cloister.jpeg` | bathroom | 0.91 | 0.88 | 0.13 |
| `nursery2.jpg` | greenhouse | 0.99 | 0.99 | 0.02 |

Both are confident and clear-margin, so no margin or entropy threshold that keeps
ordinary correct photos can flag them. The temperature T < 1 sharpens the
distribution, which is why the model is confidently wrong here.

## Conclusion

The rules are implemented (`serving/routing.py`) and off by default
(`REVIEW_MARGIN_MIN`, `REVIEW_ENTROPY_MAX`). The roadmap's acceptance line for M2
is not met. See DECISIONS D7 and D10 for the model-side attempt (M3).
