from utils.YParams import YParams
import torch
import torch.distributed as dist
import logging
import os
import argparse
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
import numpy as np
import glob


def set_seed(seed):                                       
    logging.info("Using seed {}".format(seed))                          
    import random
    random.seed(seed)                                                   
    np.random.seed(seed)                                                
    torch.manual_seed(seed)   

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0001', type=str)
    parser.add_argument("--yaml_config", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--epochs", default=0, type=int)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.epochs > 0:
        params['max_epochs'] = args.epochs

    
    print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    print('World size from Cuda: %d' % torch.cuda.device_count())
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
    else:
        params['world_size'] = torch.cuda.device_count()
    
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl', init_method='env://')

        world_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        args.gpu = local_rank

        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size // params['world_size'])
    else:
        world_rank = 0
        local_rank = 0

    #torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    
    # Logging to screen
    if params["log_to_screen"]:
        logging.basicConfig(level=logging.INFO,
                format="%(levelname)s | %(asctime)s | %(message)s")


    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    
    params['experiment_dir'] = os.path.abspath(expDir)
    ckpt_path_globstr = 'training_checkpoints/ckpt*.tar'
    best_ckpt_path = 'training_checkpoints/best_ckpt.tar'
    params['checkpoint_path_globstr'] = os.path.join(expDir, ckpt_path_globstr)
    params['best_checkpoint_path'] = os.path.join(expDir, best_ckpt_path)

    checkpoint_paths = [file for file in glob.glob(params.checkpoint_path_globstr) if os.path.isfile(file)]
    checkpoint_exists = len(checkpoint_paths) > 0

    # Determine whether to resume or start fresh
    if params.fresh_start:
        params['resuming'] = False
        if checkpoint_exists and world_rank == 0:
            logging.info("Fresh start requested. Ignoring existing checkpoint.")
    elif checkpoint_exists:
        params['resuming'] = True
        if world_rank == 0:
            logging.info("Resuming from existing checkpoint.")
    else:
        params['resuming'] = False
        if world_rank == 0:
            logging.info("No checkpoint found. Starting fresh training run.")

    params['local_rank'] = local_rank

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    
    set_seed(params["init_seed"])
    
    # Load in Trainer module
    logging.info(f'Config: {args.config}')
    if args.config == 'BASE' or args.config == 'MAE_FINETUNE' or args.config == 'SMAE_FINETUNE':
        logging.info("Importing Trainer from utils.trainer")
        from utils.trainer import Trainer
    elif args.config == 'MAE_PRETRAIN':
        logging.info("Importing Trainer from utils.mae_pretrainer")
        from utils.mae_pretrainer import Trainer
    elif args.config == 'SMAE_PRETRAIN':
        logging.info("Importing Trainer from utils.smae_pretrainer")
        from utils.smae_pretrainer import Trainer
    else:
        raise ValueError("Invalid config: {args.config}")

    trainer = Trainer(params, world_rank)

    trainer.train()

    logging.info('DONE ----- rank %d' % world_rank)
