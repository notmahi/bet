# To train.

python train.py \
  env=blockpush \
  experiment.num_cv_runs=3 \
  action_interface.action_ae.discretizer.num_bins=1,4,8,16,20,24,32,64 \
  experiment.save_subdir=reproduction/sweep_num_bins

# To evaluate.

for job in {0..7}; do
  python run_on_env.py \
    env=blockpush \
    model.load_dir="$(pwd)"/train_runs/train_blockpush/reproduction/sweep_num_bins/"$job" \
    experiment.save_subdir=reproduction/sweep_num_bins/"$job" \
    experiment.vectorized_env=True \
    experiment.async_envs=True \
    experiment.num_envs=20 \
    experiment.num_eval_eps=50 \
    experiment.device=cpu
done
