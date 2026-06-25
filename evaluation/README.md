# evaluation/

## evaluate.py

Full evaluation suite computing:
- Accuracy, AUC-ROC, F1, EER
- Confusion matrix plot
- ROC curve plot
- Score distribution histogram

```bash
python evaluation/evaluate.py \
    --data_dir /path/to/test_data \
    --checkpoint ./checkpoints/classifier/best_model.pth \
    --output_dir ./eval_results \
    --split test
```

## Output Files

```
eval_results/
├── metrics_test.json         ← all numeric metrics
├── roc_curve.png             ← ROC curve
├── confusion_matrix.png      ← confusion matrix
└── score_distribution.png    ← real vs fake score histograms
```
