# Customer segmentation with PCA and K-means

An exploratory customer segmentation project using demographics, spending,
purchasing activity, and campaign responses. The workflow compares baseline
K-means with K-means on ten principal components, then profiles a three-cluster
solution using original customer values.

## Workflow

- Explore income and total spending; impute missing income with the median.
- Derive age, tenure, and total spending.
- Robust-scale numerical features and one-hot encode categorical features.
- Exclude identifiers, constant columns, and redundant income groups from modelling.
- Use total spending rather than individual product spending for clustering.
- Inspect elbow curves and compare three- and four-cluster PCA solutions.
- Summarise customer profiles and display a normalised, annotated heatmap.

## Run

From the repository root:

```bash
python -m pip install -r clustering_project/requirements.txt
python clustering_project/cluster.py --data /path/to/customer_segmentation.csv
```

Alternatively, place `customer_segmentation.csv` beside `cluster.py` and omit
`--data`. The CSV is not distributed with this project and is ignored by Git.
Use the Customer Personality Analysis data with the column names expected by
the script, including `Income`, `Year_Birth`, `Dt_Customer`, and the six `Mnt*`
spending columns. This script expects a comma-separated CSV and customer dates
in day-month-year format.

The script prints evaluation scores and opens exploratory plots, PCA plots,
and a customer-profile heatmap. For noninteractive execution, set
`MPLBACKEND=Agg` (plots will not open or automatically be saved).

## Interpretation and limitations

Ten components retain approximately 79% of the preprocessed variance; the first
two retain approximately 41%. The three-cluster solution has a higher silhouette
score than the four-cluster solution in this run. Scores are also calculated
in the original preprocessed feature space for a common comparison with the
baseline. A two-dimensional plot alone does not establish cluster quality.

Numerical profiles use medians. Education indicators use means (proportions);
household counts use means (average counts, not proportions). Heatmap colours
are normalised separately for each feature across clusters, while annotations
show original values. Identical summaries map to zero and indicate no difference.

Age and tenure use a fixed reference year of 2025, not a verified observation
date, and tenure ignores month and day. Extreme income and birth-year values
remain in the data. Robust scaling does not remove their influence. Cluster
stability across seeds and resampled datasets has not yet been evaluated.
Cluster numbers are arbitrary labels, not rankings or validated customer types.
