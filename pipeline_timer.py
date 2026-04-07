import subprocess
import time
import sys

PYTHON = sys.executable

STEPS = [
    ("Dataset sanity check",        [PYTHON, "data/test_loader.py"]),
    ("HOG feature extraction",      [PYTHON, "preprocessing/extract_hog_features.py"]),
    ("Train baseline CNN",          [PYTHON, "training/train_baseline.py"]),
    ("Train hybrid SVM",            [PYTHON, "training/train_hybrid_svm.py"]),
    ("Evaluate models",             [PYTHON, "evaluation/evaluate_models.py"]),
    ("Robustness testing",          [PYTHON, "evaluation/robustness_test.py"]),
    ("Grad-CAM explainability",     [PYTHON, "explainability/gradcam_explain.py"]),
    ("Evaluate hybrid SVM (detail)",[PYTHON, "evaluation/evaluate_hybrid_svm.py"]),
]


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} sec"
    else:
        return f"{seconds/60:.2f} min"


def main():
    pipeline_start = time.time()
    step_times = []

    print("\n================ PIPELINE STARTED ================\n")

    for i, (name, cmd) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {name}")
        print("-" * 60)

        step_start = time.time()
        result = subprocess.call(cmd)
        step_time = time.time() - step_start

        if result != 0:
            print(f"\n[ERROR] Step failed: {name}")
            print("Pipeline stopped.")
            sys.exit(1)

        step_times.append((name, step_time))
        print(f"[DONE] {name} → {format_time(step_time)}\n")

    total_time = time.time() - pipeline_start

    # ---------------- SUMMARY ----------------
    print("\n================ PIPELINE SUMMARY ================")

    for name, t in step_times:
        print(f"{name:<35} : {format_time(t)}")

    print("-" * 50)
    print(f"{'TOTAL TIME':<35} : {format_time(total_time)}")
    print("==================================================\n")


if __name__ == "__main__":
    main()