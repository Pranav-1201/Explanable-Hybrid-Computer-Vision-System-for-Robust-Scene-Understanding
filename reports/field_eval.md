# Field evaluation

Served model, TTA on, calibrated. Policy: confidence >= 0.4780 and a home class.
72 photos: 68 labelled with one of the 67 classes, 4 matching none.
26 of them are home-class photos (the only ones that can score a room tag).

| Metric | Value | Counted over |
|---|---|---|
| Top-1 (67-class) | 0.794 | 68 labelled photos |
| Home-class top-1 | 0.846 | 26 home photos |
| Auto-tagged | 24 | 72 photos |
| Tag precision | 0.875 (95% CI 0.69 to 0.96) | 24 auto-tagged |
| Home tag recall | 0.808 (95% CI 0.62 to 0.91) | 26 home photos |
| Non-home photos given a home tag | 2 | 46 non-home or OOD photos |
| Reviewed | 48 (0.667) | 72 photos |
| Review precision | 0.979 | 48 reviewed |
| Review reasons | {'low_confidence': 7, 'out_of_scope': 41} | |

With this few photos the intervals above are wide: a one-photo change moves a
proportion by several points and is not evidence of a better or worse model.
