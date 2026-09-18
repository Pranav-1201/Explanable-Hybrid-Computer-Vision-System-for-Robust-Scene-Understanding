import sys; sys.path.insert(0, '.')
import os; os.environ['DRY_RUN'] = '1'
from training.train_phase2 import *
model = CNNBaseline(NUM_CLASSES, BACKBONE)
opt   = get_optimizer(model, BASE_LR)
print('✓ train_phase2.py imports and builds without error'.encode('utf-8').decode('cp1252', 'ignore'))
print(f'  Backbone: {BACKBONE}')
print(f'  Optimizer param groups: {len(opt.param_groups)}')
for g in opt.param_groups:
    print(f'    lr={g["lr"]:.2e}  params={len(g["params"])}')
