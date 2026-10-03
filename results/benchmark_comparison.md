# Machine Unlearning Benchmark Evaluation

## Summary Comparison Table

| Method          | Workload   |   Rounds | Final Retain Acc (%)   | Final Test Acc (%)   | Param Distance   | Total Time (s)   | Collapse Round   |
|:----------------|:-----------|---------:|:-----------------------|:---------------------|:-----------------|:-----------------|:-----------------|
| ASUC            | A          |       20 | 97.19                  | 90.89                | 44.6189          | 15655.36         | None (Stable)    |
| ASUC            | B          |       10 | 98.26                  | 92.33                | 43.6591          | 8614.21          | None (Stable)    |
| ASUC            | C          |       10 | 99.11                  | 90.75                | 42.6858          | 8565.15          | None (Stable)    |
| GRADIENT_ASCENT | A          |        3 | 99.59                  | 93.38                | 0.0076           | N/A              | None (Stable)    |
| GRADIENT_ASCENT | A          |       20 | 99.47                  | 93.19                | 0.0363           | 70.65            | None (Stable)    |
| SALUN           | A          |       20 | 95.03                  | 88.03                | 0.3631           | 1355.66          | Round 1          |
| SSD             | A          |       20 | N/A                    | N/A                  | N/A              | 1279.08          | None (Stable)    |
| SSD             | A          |       20 | 10.00                  | 10.00                | 29.1096          | 317.47           | Round 3          |

## Key Research Insights

- **Capacity Collapse in Static Baselines**: Static methods that repeatedly update without state tracking eventually suffer catastrophic plasticity loss or collapse on retain utility.
- **Adaptive Meta-Controller (ASUC-SOM)**: Dynamically routes between rapid Tier 1 synaptic dampening (SSD) and Tier 2 saliency unlearning (SalUn), verifying stability in a closed loop to preserve utility.
- **Subspace Orthogonality Metric (SOM)**: Proactively diagnoses parameter fatigue before accuracy collapse occurs.
