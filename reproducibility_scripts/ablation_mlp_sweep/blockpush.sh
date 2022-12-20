# To train.

python train.py \
  env=blockpush \
  experiment.num_cv_runs=1 \
  model=mlp \
  sweeper=optuna \
  'experiment.lr=tag(log, interval(1e-5, 1e-1))' \
  'experiment.grad_norm_clip=choice(inf, 1)' \
  'experiment.weight_decay=0.01,0.05,0.1' \
  'model.hidden_dim=72,128,144' \
  'model.hidden_depth=4,6,8' \
  'model.batchnorm=choice(True, False)' \
  hydra.sweeper.n_trials=25 \
  experiment.save_subdir=reproduction/ablation_mlp_sweep
