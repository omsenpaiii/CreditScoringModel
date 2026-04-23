# SIT723 Detailed Progress Report

Prepared on 23 April 2026

## Summary

This report documents the current SIT723 project status, finalised methodology, completed experimental design, preliminary benchmark experiments, and the current PaddleOCR blocker under active debugging.

## Best pilot results

- German Credit best run: logistic_regression with class_weight and ROC-AUC 0.831
- Lending Club sample best run: logistic_regression with class_weight and ROC-AUC 0.722

## OCR status

- OCR pilot environment setup is partially complete but the fine-tuning run has not completed yet.
- Current debugging focus: the PaddleOCR bootstrap sequence progressed through package installation, repository clone, synthetic dataset generation, and pretrained archive download, but the run did not yet reach a stable saved fine-tuned checkpoint.
