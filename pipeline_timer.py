# pipeline_timer.py
# ============================================================
# Pipeline Timer Wrapper (Windows-safe, venv-safe)
# ============================================================

import subprocess
import time
import sys

PYTHON = sys.executable  # 🔥 THIS IS THE KEY FIX

STEPS = [
    ("[1/8] Dataset sanity check", [PYTHON, "data/test_loader.py"]),
    ("[2/8] HOG feature extraction", [PYTHON, "preprocessing/extract_hog_features.py"]),
    ("[3/8] Train baseline CNN", [PYTHON, "training/train_baseline.py"]),
    ("[4/8] Train hybrid model", [PYTHON, "training/train_hybrid.py"]),
    ("[5/8] Evaluate models", [PYTHON, "evaluation/evaluate_models.py"]),
    ("[6/8] Robustness testing", [PYTHON, "evaluation/robustness_test.py"]),
    ("[7/8] Grad-CAM explainability", [PYTHON, "explainability/gradcam_explain.py"]),
]

def main():
    pipeline_start = time.time()
    print("\n================ PIPELINE STARTED ================\n")

    for name, cmd in STEPS:
        print(name)
        print("-" * 60)

        step_start = time.time()
        result = subprocess.call(cmd)
        step_time = time.time() - step_start

        if result != 0:
            print("\nERROR: Pipeline stopped.")
            sys.exit(1)

        print(f"[DONE] Step time: {step_time:.2f} seconds\n")

    total_time = time.time() - pipeline_start

    print("================ PIPELINE SUMMARY ================")
    print(f"Total Pipeline Time : {total_time:.2f} seconds")
    print(f"Total Pipeline Time : {total_time/60:.2f} minutes")
    print("==================================================")

if __name__ == "__main__":
    main()