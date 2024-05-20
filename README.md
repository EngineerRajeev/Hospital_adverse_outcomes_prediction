# Predictive Analysis of Adverse Hospital Outcomes

## Overview

This repository contains the code and data used for predicting in-hospital mortality, readmission, and prolonged length of stay using the XRAI framework. The analysis leverages historical electronic health records (EHR) and employs various machine learning models.

## Repository Structure.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│ ├── HO30DMPM.ipynb
│ ├── ReadmissionPM.ipynb
│ ├── PLOS_Lastepisode.ipynb
│ ├── XRAI_PrePro.ipynb
├── data/
│ ├── Readm_stats.csv
│ ├── PLOS_stats.csv
│ ├── 30Dmortality_stats.csv
│ ├── 30Dmort_start_stats.csv
├── scripts/
│ ├── preprocessing.py
│ ├── train_model.py
│ ├── evaluate_model.py
├── results/
│ ├── evaluation_metrics.csv
│ ├── confusion_matrix.png
│ ├── feature_importance.png
