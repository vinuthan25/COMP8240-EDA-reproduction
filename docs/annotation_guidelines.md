# Labelling guidelines: grocery-app reviews

These are the rules used to label the 199 Google Play reviews (Woolworths and Coles apps).

## Labels
Each review gets one label, judged from the **text alone**. The star rating was hidden while labelling.

| Label | Code | Meaning |
|---|---|---|
| negative | 0 | The reviewer is mainly complaining. A mild complaint still counts as negative. |
| positive | 1 | The reviewer is mainly happy. |
| mixed | 2 | Praise and complaint together, or no real opinion. |

If a review is mixed, the label is *mixed* unless it ends with a clear verdict (for example "useless" or "I'll shop elsewhere"), in which case that side wins.

## Illustrative examples (invented, not from the dataset)
- "Easy ordering and the delivery was always on time." -> positive
- "Crashes every time I get to checkout." -> negative
- "Good prices but the app is slow." -> mixed

## Procedure
1. The reviews were sampled evenly across star ratings (40 per star), shuffled, and exported **without** their star ratings.
2. An AI assistant that did not see the star ratings suggested a first-pass label for every review, using these definitions.
3. The author reviewed all labels and changed five of the 199.
4. The labels were checked against labels derived from the star ratings (1-2 stars negative, 4-5 positive, 3 stars excluded): 91.9% agreement, Cohen's kappa 0.84 (149 reviews). This is agreement with a proxy, not agreement between two human annotators.

## Lesson learned
During review the author used a different numeric code (2 = negative, 0 = neutral) from the one in these guidelines, which made 81 labels look changed when only five were real disagreements. Labelling with words instead of digits would have avoided this.

## Judge rubric
The separate LLM-as-judge rubric for the edited texts is in `judge/judge_rubric.txt`.
