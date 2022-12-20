# To train.

python train.py \
  env=kitchen \
  experiment.num_cv_runs=3 \
  action_interface.action_ae.discretizer.num_bins=1,3,4,5,16,60,64,68 \
  experiment.save_subdir=reproduction/sweep_num_bins

# To evaluate.

for job in {0..7}; do
  python run_on_env.py \
    env=kitchen \
    model.load_dir="$(pwd)"/train_runs/train_kitchen/reproduction/sweep_num_bins/"$job" \
    experiment.save_subdir=reproduction/sweep_num_bins/"$job" \
    experiment.vectorized_env=True \
    experiment.async_envs=True \
    experiment.num_envs=20 \
    experiment.num_eval_eps=50 \
    experiment.device=cpu
done
