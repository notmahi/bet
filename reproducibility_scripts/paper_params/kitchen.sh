# To train.
python train.py \
  env=kitchen \
  experiment.num_cv_runs=5 \
  experiment.save_subdir=reproduction/paper_params

# To evaluate.

python run_on_env.py \
  env=kitchen \
  model.load_dir="$(pwd)"/train_runs/train_kitchen/reproduction/paper_params/0 \
  experiment.save_subdir=reproduction/paper_params/0 \
  experiment.vectorized_env=True \
  experiment.async_envs=True \
  experiment.num_envs=20 \
  experiment.num_eval_eps=50 \
  experiment.device=cpu
