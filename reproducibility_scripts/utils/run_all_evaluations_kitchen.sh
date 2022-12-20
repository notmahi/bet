for exp_dir in train_runs/train_kitchen/reproduction/* ; do
    for job_dir in "$exp_dir"/* ; do
      if [ -d "$job_dir" ]; then
        exp=$(dirname "$job_dir")
        save_dir=$(basename "$exp")/$(basename "$job_dir")
        echo "$job_dir" "->" "$save_dir"

        python run_on_env.py \
        env=kitchen \
        model.load_dir="$(pwd)"/"$job_dir" \
        experiment.save_subdir=reproduction/"$save_dir" \
        experiment.vectorized_env=True \
        experiment.async_envs=True \
        experiment.num_envs=20 \
        experiment.num_eval_eps=50 \
        experiment.device=cpu
      fi
    done
done
