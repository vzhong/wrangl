python train.py --multirun hydra/launcher=slurm hydra.launcher.partition=learnlab wandb.enable=true model.name=multi env.name=rtfm_train_s1-v0 suffix=1,2,3,4,5,6,7,8 project=silg-moolib

python train.py --multirun hydra/launcher=slurm hydra.launcher.partition=learnlab wandb.enable=true env.name=alfworld_train-v0 model.name=multi_rank batch_size=32 virtual_batch_size=32 actor_batch_size=64 suffix=1,2,3,4,5,6,7,8 project=silg-moolib

python train.py --multirun hydra/launcher=slurm hydra.launcher.partition=learnlab wandb.enable=true env.name=td_segs_train-v0 model.name=multi_nav_emb batch_size=3 virtual_batch_size=3 actor_batch_size=6 model.num_film=3 model.drnn=100 model.drep=200 model.demb=30 unroll_length=64 model.stateful=true num_actor_cpus=6 num_actor_batches=1 suffix=1,2,3,4,5,6,7,8 project=silg-moolib


