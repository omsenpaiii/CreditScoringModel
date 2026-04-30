# AI-Powered Credit Scoring and Document OCR System for Loan Risk Assessment

Prepared for CO-OP Project at Industry (Module-2)

## Team

- Om Tomar (2210994882)
- Sahajpal Singh (2210994831)
- Ridhima Chopra (2210994826)
- Ishpreet Kaur (2210990424)

## Project Metadata

- Project type: Copyright
- Supervisor/Mentor: Lalit Sharma
- Department: Computer Science and Engineering, Chitkara University, Punjab
- Current status: Submission-ready package with runnable code, report, presentation, and IPR placeholder

## Abstract

This project presents an AI-powered loan-risk assessment workflow that combines tabular credit scoring with a real-data document OCR extension. The credit-scoring module benchmarks Logistic Regression, Random Forest, Gradient Boosting, and Multi-Layer Perceptron models on German Credit and Lending Club sample data. The OCR module prepares ICDAR2019 SROIE and FUNSD scanned-document annotations for PaddleOCR evaluation and fine-tuning.

## Best Credit-Scoring Results

- German Credit: Logistic Regression, class_weight, ROC-AUC 0.831, PR-AUC 0.713, Recall 0.778, F1 0.673
- Lending Club Sample: Logistic Regression, class_weight, ROC-AUC 0.722, PR-AUC 0.360, Recall 0.682, F1 0.430

## OCR Module Implementation

- PaddleOCR workflow: document upload, preprocessing, recognition, field extraction, and review queue.
- Real OCR data basis: ICDAR2019 SROIE scanned receipts and FUNSD noisy scanned forms.
- Current OCR status: Real SROIE/FUNSD OCR pipeline status: prepared_real_dataset. Prepared recognition crops: train: 1200, val: 300, test: 300.
- Evaluation method: compare OCR outputs against held-out public scanned-document labels, then repeat on private borrower documents after privacy review and annotation.

## Copyright Appendix

- Original work claimed: source code, configurations, report text, diagrams, integration workflow, generated figures, and final submission packaging.
- Third-party work not claimed: PaddleOCR/PaddlePaddle, public datasets, pretrained weights, and external Python libraries.
- Form XIV support notes are documented in `Source code/docs/COPYRIGHT_PACKAGE.md`.

## Evaluation Readiness

- Submission / Filing Status: IPR proof folder created with copyright filing instructions and placeholder.
- Target Platform Compliance: GitHub repository follows the required folder structure and contains runnable code.
- Revision and Documentation: final report, PPT, README, figures, templates, and reproducible generator are included.
- Supervisor Meetings: progress updates and feedback handling are documented for final review.
