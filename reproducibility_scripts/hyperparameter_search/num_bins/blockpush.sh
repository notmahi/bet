# To train
python train.py env=blockpush experiment.num_cv_runs=3 action_interface.action_ae.discretizer.num_bins=1,4,8,16,20,24,32,64

# To evaluate