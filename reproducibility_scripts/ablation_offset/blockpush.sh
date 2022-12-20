# To train.

python train.py \
  env=blockpush \
  experiment.num_cv_runs=3 \
  model.predict_offsets=False \
  experiment.save_subdir=reproduction/ablation_offset

# To evaluate.

python run_on_env.py \
  env=blockpush \
  model.load_dir="$(pwd)"/train_runs/train_blockpush/reproduction/ablation_offset/0 \
  experiment.save_subdir=reproduction/ablation_offset/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=True \
  experiment.num_envs=20 \
  experiment.num_eval_eps=50 \
  experiment.device=cpu
