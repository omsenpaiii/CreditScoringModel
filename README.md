# AI-Powered Credit Scoring and Document OCR System for Loan Risk Assessment

## Team Details

| Name | Roll Number |
|---|---:|
| Om Tomar | 2210994882 |
| Sahajpal Singh | 2210994831 |
| Ridhima Chopra | 2210994826 |
| Ishpreet Kaur | 2210990424 |

## Project Details

- **Project type:** Copyright
- **Supervisor/Mentor:** Lalit Sharma
- **Department:** Computer Science and Engineering, Chitkara University
- **Current status:** Final CO-OP submission package prepared for external evaluation

## Repository Structure

| Folder | Contents |
|---|---|
| `IPR Submission Proof` | Placeholder for the copyright submission form and screenshot. |
| `Report and PPT` | Final report, presentation, original templates, markdown source, and supporting figures. |
| `Source code` | Runnable Python source code, configs, tests, data samples, third-party OCR resources, and saved experiment artifacts. |

## Highlights

- Credit-scoring benchmark on German Credit and Lending Club sample datasets.
- Models: Logistic Regression, Random Forest, Gradient Boosting, and Multi-Layer Perceptron.
- Saved metrics: ROC-AUC, PR-AUC, precision, recall, F1-score, accuracy, confusion matrices, ROC/PR curves, and feature-importance plots.
- PaddleOCR workflow included for borrower-document intake, preprocessing, recognition, field extraction, and review.
- Real OCR preparation pipeline added for ICDAR2019 SROIE and FUNSD scanned-document datasets, with measured PaddleOCR baseline evaluation support.
- Best saved results: German Credit ROC-AUC `0.831`, Lending Club sample ROC-AUC `0.722`.

## Final Evaluation Readiness

- **Submission / Filing Status:** `IPR Submission Proof` is prepared with copyright filing instructions and placeholder.
- **Target Platform Compliance:** GitHub structure follows the required folders and includes runnable code.
- **Revision, Feedback Handling, Documentation:** final report, PPT, README, figures, templates, and reproducible generator are included.
- **Supervisor Meetings and Progress Updates:** final documentation records progress, implementation status, and feedback-driven restructuring.

## Run Instructions

From the repository root:

```bash
cd "Source code"
python3 -m pip install -r requirements.txt
PYTHONPATH="src" python3 tests/smoke_test.py
```

Run the full credit-scoring benchmark:

```bash
cd "Source code"
PYTHONPATH="src" python3 run_credit_benchmark.py --config configs/credit/benchmark_suite.json
```

Prepare real OCR data and run the pretrained PaddleOCR baseline:

```bash
cd "Source code"
PYTHONPATH="src" python3 run_ocr_experiment.py --config configs/ocr/real_ocr_finetune.json
```

Optional CPU fine-tuning smoke run:

```bash
cd "Source code"
PYTHONPATH="src" python3 run_ocr_experiment.py --config configs/ocr/real_ocr_finetune_train_cpu.json
```

Regenerate the final report and presentation:

```bash
PYTHONPATH="Source code/src" python3 "Source code/scripts/generate_final_submission.py"
```

## Final Evaluation Files

- `Report and PPT/Final_COOP2_Project_Report.docx`
- `Report and PPT/Final_COOP2_External_Presentation.pptx`
- `Report and PPT/Final_COOP2_Project_Report.md`
- `Report and PPT/figures/`

## IPR Note

This repository is structured for a copyright-targeted submission. The actual copyright application form and submission screenshot must be added to `IPR Submission Proof` after the team completes the official filing process. The team copyright claim covers original code, diagrams, documentation, report, presentation, configurations, and integration workflow; third-party libraries, public datasets, and pretrained weights are not claimed as original work.
