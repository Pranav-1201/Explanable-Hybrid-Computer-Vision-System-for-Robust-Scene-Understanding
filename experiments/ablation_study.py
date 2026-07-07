# experiments/ablation_study.py
"""
Ablation study plan for the Hybrid CV System.
Run after all models are trained.

Experiments:
  A. CNN only vs CNN + HOG fusion       — Does HOG add value?
  B. Places365 vs ImageNet pre-training — Does domain match matter?
  C. With EMA vs without EMA            — Is EMA worth it?
  D. With TTA vs single-crop            — Inference time tradeoff?
  E. Temperature scaling effect         — Calibration value?
"""

ABLATION_PLAN = [
    {
        'name':     'HOG contribution',
        'variable': 'hog_features',
        'baseline': 'fusion_best.pth (CNN+HOG)',
        'ablated':  'ablate_hog() method',
        'metric':   'test_accuracy',
        'expected_delta': '+2 to +4%',
    },
    {
        'name':     'Places365 vs ImageNet',
        'variable': 'pretrain_dataset',
        'baseline': 'phase2_best.pth (Places365)',
        'ablated':  'resnet50_imagenet fallback run',
        'metric':   'test_accuracy',
        'expected_delta': '+8 to +12%',
    },
    {
        'name':     'EMA benefit',
        'variable': 'ema_decay',
        'baseline': 'EMA model (ema_state in checkpoint)',
        'ablated':  'raw model (model_state in checkpoint)',
        'metric':   'val_accuracy',
        'expected_delta': '+0.5 to +1.5%',
    },
    {
        'name':     'TTA benefit',
        'variable': 'inference_crops',
        'baseline': 'tta_predict (3 transforms)',
        'ablated':  'single_predict',
        'metric':   'test_accuracy',
        'expected_delta': '+2 to +4%',
    },
]

def log_result(experiment_name, baseline_acc, ablated_acc, notes=''):
    import json, datetime
    result = {
        'timestamp':     datetime.datetime.now().isoformat(),
        'experiment':    experiment_name,
        'baseline_acc':  baseline_acc,
        'ablated_acc':   ablated_acc,
        'delta':         round(baseline_acc - ablated_acc, 4),
        'notes':         notes,
    }
    with open('results/ablation_results.jsonl', 'a') as f:
        f.write(json.dumps(result) + '\n')
    print(f"[ABLATION] {experiment_name}: {baseline_acc*100:.2f}% vs "
          f"{ablated_acc*100:.2f}% (delta={result['delta']*100:+.2f}%)")
    return result

if __name__ == '__main__':
    print("Ablation study plan:")
    for i, exp in enumerate(ABLATION_PLAN, 1):
        print(f"\n  {i}. {exp['name']}")
        print(f"     Variable : {exp['variable']}")
        print(f"     Baseline : {exp['baseline']}")
        print(f"     Ablated  : {exp['ablated']}")
        print(f"     Expected : {exp['expected_delta']}")
