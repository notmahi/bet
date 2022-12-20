# Train with params in the paper.

python train.py \
  env=kitchen \
  experiment.num_cv_runs=3 \
  model=mlp \
  experiment.save_subdir=reproduction/ablation_mlp/paper

# Eval with params in the paper.

python run_on_env.py \
  env=kitchen \
  model.load_dir="$(pwd)"/train_runs/train_kitchen/reproduction/ablation_mlp/paper/0 \
  experiment.save_subdir=reproduction/ablation_mlp/paper/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=True \
  experiment.num_envs=20 \
  experiment.num_eval_eps=50 \
  experiment.device=cpu

# Train with best sweep params.
## Best one was:
## train_runs/train_kitchen/reproduction/ablation_mlp_sweep/<SOMEWHERE>
## Retrain with 3 seeds.

python train.py \
  env=kitchen \
  experiment.num_cv_runs=3 \
  model=mlp \
  experiment.lr=x \
  experiment.grad_norm_clip=x \
  experiment.weight_decay=x \
  model.hidden_dim=x \
  model.hidden_depth=x \
  model.batchnorm=x \
  experiment.save_subdir=reproduction/ablation_mlp/sweep_best

# Eval with best sweep params.

python run_on_env.py \
  env=kitchen \
  model.load_dir="$(pwd)"/train_runs/train_kitchen/reproduction/ablation_mlp/sweep_best/0 \
  experiment.save_subdir=/reproduction/ablation_mlp/sweep_best/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=True \
  experiment.num_envs=20 \
  experiment.num_eval_eps=50 \
  experiment.device=cpu



