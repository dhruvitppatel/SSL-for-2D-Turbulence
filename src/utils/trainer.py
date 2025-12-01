from models.vision_transformer import ViT
from utils.data_loaders import get_dataloader
import os
from collections import OrderedDict
import time
import wandb
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils.diagnostics import grad_norm, grad_max, log_input_target_prediction
import glob
from natsort import natsorted
from pathlib import Path
import numpy as np
#from torch.profiler import profile, record_function, ProfilerActivity


#torch.backends.cuda.enable_flash_sdp(True)

class Trainer():
    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.early_stop_epoch = params['early_stop_epoch'] - 1 if 'early_stop_epoch' in params else None
        self.device_ID = dist.get_rank()

        logging.info('rank %d, begin data loader init' % world_rank)
        print(params)

        self.iteration_metrics = []
        self.valid_loss = 0
        self.mean_epoch_loss = 0

        # Construct training/validation dataloaders
        self.train_dataloader, self.train_dataset, self.train_sampler = get_dataloader(data_dir=params["data_dir"],
                                                                                       file_range=params["train_file_range"],
                                                                                       target_step=params["target_step"],
                                                                                       train_tendencies=params["train_tendencies"],
                                                                                       batch_size=params["batch_size"],
                                                                                       train=True,
                                                                                       distributed=dist.is_initialized(),
                                                                                       num_frames=params["num_frames"],
                                                                                       num_out_frames=params["num_out_frames"],
                                                                                       num_workers=params["num_workers"],
                                                                                       pin_memory=params["pin_memory"],
                                                                                       forward_step=params['forward_step'])

        self.valid_dataloader, self.valid_dataset = get_dataloader(data_dir=params["data_dir"],
                                                                   file_range=params["valid_file_range"],
                                                                   target_step=params["target_step"],
                                                                   train_tendencies=params["train_tendencies"],
                                                                   batch_size=params["batch_size"],
                                                                   train=False,
                                                                   distributed=dist.is_initialized(),
                                                                   num_frames=params["num_frames"],
                                                                   num_out_frames=params["num_out_frames"],
                                                                   num_workers=params["num_workers"],
                                                                   pin_memory=params["pin_memory"],
                                                                   forward_step=params['forward_step'])
        logging.info("data loader initialized")

 
        # Set up logging with wandb - only on main process
        if params.log_to_wandb:

            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = os.path.dirname(self.params.checkpoint_path_globstr)
            if not os.path.exists(checkpoint_dir) and world_rank == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            wandb_id_path = Path(checkpoint_dir) / "wandb_id.txt"

            if params.resuming and wandb_id_path.exists():               # Resume run
                run_id  = wandb_id_path.read_text().strip()
                resume  = "allow"
                logging.info(f"Resuming wandb run with id: {run_id}")
            else:                                    # fresh run
                run_id  = wandb.util.generate_id()
                if world_rank == 0:
                    wandb_id_path.write_text(run_id)
                resume  = None
                logging.info(f"Starting fresh wandb run with id: {run_id}")

            wandb.init(config=params, name=params.name, group=params.group, project=params.project,
                      id=run_id, resume=resume)
            
            wandb.define_metric("epoch")
            epoch_metrics = ['lr', 'train_loss', 'val_loss']
            for metric in epoch_metrics:
                wandb.define_metric(metric, step_metric="epoch")
            
            self.wandb_table = wandb.Table(columns=['Id', 'Input', 'Target', 'Prediction', 'Target-Prediction'])
            

        # Construct model
        self.model = ViT(
            img_size=params["img_size"],
            patch_size=params["patch_size"],
            num_frames=params["num_frames"],
            tubelet_size=params["tubelet_size"],
            in_chans=params["in_chans"],
            encoder_embed_dim=params["encoder_embed_dim"],
            encoder_depth=params["encoder_depth"],
            encoder_num_heads=params["encoder_num_heads"],
            decoder_embed_dim=params["decoder_embed_dim"],
            decoder_depth=params["decoder_depth"],
            decoder_num_heads=params["decoder_num_heads"],
            mlp_ratio=params["mlp_ratio"],
            num_out_frames=params["num_out_frames"],
            patch_recovery=params["patch_recovery"],
            checkpointing=params["checkpointing"])

        # If finetuning, load pre-trained model weights
        if params["mae_finetune"]:
            checkpoint_model_temp = torch.load(params["mae_finetune_fp"], map_location='cpu', weights_only=False)['model_state']
            checkpoint_model = {}
            for key, val in checkpoint_model_temp.items():
                key_new = key[7:]                  # Removing 'module.' that is appended before each key by DDP
                checkpoint_model[key_new] = val

            print(f"Load pre-trained checkpoint from: {params['mae_finetune_fp']}")
            print(f"Pre-trained checkpoint_model.keys(): {checkpoint_model.keys()}")
            state_dict = self.model.state_dict()
            #for k in ['head.weights', 'head.bias']:
            #    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            #        print(f"Removing key {k} from pretrained checkpoint.")
            #        del checkpoint_model[k]
            for key, val in state_dict.items():
                if key in checkpoint_model.keys() and val.shape != checkpoint_model[key].shape:
                    print(f'Removing key {key} from pretrained checkpoint due to shape mismatch.')
                    del checkpoint_model[key]

            # Drop decoder from pretrained checkpoint
            for key, val in state_dict.items():
                for layer in params['drop_layers']:
                    if layer in key:
                        print(f'Removing DECODER key {key} from pretrained checkpoint.')
                        del checkpoint_model[key]
        
            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # Freeze layers
            for name, module in self.model.named_parameters():
                for freeze_layer in params["freeze_layers"]:
                    if freeze_layer in name:
                        print(f'Freezing {name}')
                        module.requires_grad = False

            print(f'model.state_dict().keys(): {self.model.state_dict().keys()}')

        # Send model to device
        self.model.to(self.device)


        # Watch model gradients with wandb
        if params.log_to_wandb:
            wandb.watch(self.model)


        # Set optimizer
        if params.optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], fused=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])


        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[params.local_rank],
                                                 output_device=[params.local_rank],
                                                 find_unused_parameters=True)
       

        # Set learning rate scheduluer
        if params["scheduler"] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=float(params['factor']), patience=int(params['patience']), 
                                                                        cooldown=int(params['cooldown']), mode='min')
        elif params["scheduler"] == 'CosineAnnealingLR':
            # Adjustig max epochs input to CosineAnnealingLR scduler if using warmup
            if params['warmup']:
                # Subtract warmup epochs from max_epochs
                T_max = int(params["max_epochs"]) - int(params['warmup_totaliters'])
            else:
                T_max = int(params["max_epochs"]) 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=float(params['lr_min']))
        elif params["scheduler"] == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=int(params["T_0"]), 
                                                                                  T_mult=int(params["T_mult"]), eta_min=float(params['lr_min']))
        else:
            self.scheduler = None

        # Chain warmup with main scheduler using SequentialLR
        if params.warmup and self.scheduler is not None:
            warmuplr = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=params.warmup_startfactor,
                                                         total_iters=params.warmup_totaliters)
            main_scheduler = self.scheduler
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmuplr, main_scheduler],
                milestones=[params.warmup_totaliters]
            )


        if params.resuming:
            ckpt_dir = Path(self.params.checkpoint_path_globstr).parent
            checkpoint_path = str(ckpt_dir / "ckpt.tar")
            print(f'RESTORE CKPT: {checkpoint_path}')
            self.restore_checkpoint(checkpoint_path) # Restore model and optimizer state and learning rate scheduler state if any
        else:
            logging.info("Starting fresh training run")
            self.iters = 0
            self.best_valid_loss = 1.e6
            self.startEpoch = 0

        self.epoch = self.startEpoch

        if params["log_to_screen"]:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
            print("Number of trainable model parameters: {}".format(self.count_parameters()))


    def count_parameters(self):
        if dist.is_initialized():
            return sum(p.numel() for p in self.model.module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def train(self):
        # Iterate over all epocs
        if self.params["log_to_screen"]:
            logging.info("Starting training loop ...")

        early_stopping_counter = 0
        early_stop_epoch_triggered = False

        for epoch in range(self.startEpoch, self.params["max_epochs"]):
            
            if self.early_stop_epoch is not None and epoch > self.early_stop_epoch:
                if self.params.log_to_screen:
                    logging.info(f'Completed early stop epoch {self.early_stop_epoch}. Terminateing training.')
                early_stop_epoch_triggered = True
                break

            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)


            start = time.time()

            tr_time, data_time, train_logs = self.train_one_epoch() # Train one epoch
            valid_time, valid_logs = self.validate_one_epoch()

            # Adjust lr rate schedule if using
            if self.params["scheduler"] == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params["scheduler"] == 'CosineAnnealingLR' or self.params["scheduler"] == 'CosineAnnealingWarmRestarts':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    break

            if self.params.log_to_wandb and self.device_ID == 0:

                for i, metrics in enumerate(self.iteration_metrics):
                    wandb.log(metrics, step=metrics['iteration'])

                # Then log the epoch summary
                epoch_summary = {
                    "epoch": self.epoch,
                    "epoch_loss": self.mean_epoch_loss,
                    "epoch_loss_best": self.best_loss,
                    "valid_loss": self.valid_loss,
                    "valid_loss_best": self.best_valid_loss,
                    "lr": self.optimizer.param_groups[0]['lr']
                    }

                wandb.log(epoch_summary, step=self.iters)


                # wandb.log({'lr': lr, 'epoch': self.epoch}, step=self.epoch)


            # Early stopping logic should be outside of world_rank check
            if valid_logs["valid_loss"] <= self.best_valid_loss:
               self.best_valid_loss  = valid_logs['valid_loss']
               early_stopping_counter = 0
            else:
               early_stopping_counter += 1

            if self.world_rank == 0:
               if self.params.save_checkpoint:
                  ckpt_dir = Path(self.params.checkpoint_path_globstr).parent
                  checkpoint_path_out = str(ckpt_dir / "ckpt")
                  self.save_checkpoint(checkpoint_path_out + f'.tar')
                  if valid_logs["valid_loss"] <= self.best_valid_loss:
                     self.save_checkpoint(self.params.best_checkpoint_path)
                  if (self.epoch+1) in self.params.ckpt_epoch_list:
                      logging.info(f"Saving checkpoint at epoch {self.epoch+1}")
                      self.save_checkpoint(checkpoint_path_out + f'_{self.epoch+1}.tar')


            if self.params["log_to_screen"]:
                logging.info("Time taken for epoch {} is {} sec".format(epoch+1, time.time()-start))
                logging.info("Train loss: {}. Valid loss: {}".format(train_logs['train_loss'], valid_logs["valid_loss"]))

                if self.params.early_stopping:
                   logging.info(f'Early stopping counter: {early_stopping_counter} out of {self.params.early_stopping_patience}')

            if self.params.early_stopping and early_stopping_counter >= self.params.early_stopping_patience:
               if self.params.log_to_screen:
                  logging.info('Early stopping triggered. Terminating training.')
               break


        # After training loop ends
        if self.params.log_to_wandb:
          if self.world_rank == 0:
             self.log_all_plots_wandb()
        
        if self.params.log_to_screen:
           if early_stop_epoch_triggered:
              logging.info(f'Training finished early at epoch {self.early_stop_epoch} due to early_stop_epoch setting.')
           else:
              logging.info('Completed all epochs. Training finished normally.')   


        logging.info("----- DONE -----")

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()


        total_iterations = len(self.train_dataloader)
        losses = []
        diagnostic_logs = {}

        if self.params.log_to_wandb and self.device_ID == 0:
            self.iteration_metrics = []

        for i, data in enumerate(self.train_dataloader):

            self.iters += 1
            data_start = time.time()
            inputs, labels = data[0].to(self.device, dtype=torch.float32), data[1].to(self.device, dtype=torch.float32)
            data_time += time.time() - data_start

            tr_start = time.time()

            self.model.zero_grad()
            self.optimizer.zero_grad(set_to_none=True)

            # # Profile
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            #    with record_function("model inference"):
            
            if self.params['integrator'] == 'E1':
                outputs = self.E1_integrator(inputs, train=True)
            elif self.params['integrator'] == 'RK2':
                outputs = self.RK2_integrator(inputs, train=True)
            elif self.params['integrator'] == 'RK4':
                outputs = self.RK4_integrator(inputs, train=True)
                print('************* RK4 *************')
            else:
                outputs = self.model(inputs, train=True)
            

            if dist.is_initialized():
                loss = self.model.module.forward_loss(labels, outputs)

                if self.params['spectral_loss']:
                    loss += self.model.module.spectral_loss(labels, outputs, self.params['spectral_loss_weight'], self.params['spectral_loss_threshold_wavenumber'])
            else:
                loss = self.model.forward_loss(labels, outputs)

                if self.params['spectral_loss']:
                    loss += self.model.spectral_loss(labels, outputs, self.params['spectral_loss_weight'], self.params['spectral_loss_threshold_wavenumber'])

            losses.append(loss.item())
            loss.backward()

            # Apply gradient clipping if specified
            if self.params.get('clip_grad_norm') is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip_grad_norm'])

            self.optimizer.step()

            tr_time += time.time() - tr_start

            with torch.no_grad():
                # Computations within the track wont be tracked
                # Loggings in Weights and biases
                if self.params.diagnostic_logs:
                    diagnostic_logs['batch_grad_norm'] = torch.tensor([grad_norm(self.model)]).to(self.device)
                    diagnostic_logs['batch_grad_max'] = torch.tensor([grad_max(self.model)]).to(self.device)
                    diagnostic_logs['train_batch_loss'] = loss
                    if dist.is_initialized():
                        for key in sorted(diagnostic_logs.keys()):
                            if key == 'batch_grad_max':
                                grad_max_tensor = torch.zeros(dist.get_world_size(), dtype=torch.float32, device=self.device)
                                dist.all_gather_into_tensor(grad_max_tensor, diagnostic_logs[key])
                                diagnostic_logs[key] = torch.max(grad_max_tensor)
                            else:
                                dist.all_reduce(diagnostic_logs[key].detach())
                                diagnostic_logs[key] = float(diagnostic_logs[key] / dist.get_world_size())
                    # if self.params.log_to_wandb:
                    #     wandb.log(diagnostic_logs, step=self.iters)

                            # Store metrics for this iteration
                                        # Calculate maximum gradient value across all parameters
                if self.params.log_to_wandb and self.device_ID == 0:

                    batch_grad_norm = grad_norm(self.model)
                    batch_grad_max = grad_max(self.model)
                    iter_metrics = {
                        "iteration": self.iters,
                        "batch_loss": loss.item(),
                        "batch_grad_norm": batch_grad_norm,
                        "batch_grad_max": batch_grad_max,
                    }

                    self.iteration_metrics.append(iter_metrics)

            
            torch.cuda.empty_cache()

            self.mean_epoch_loss = np.mean(losses)

            if self.epoch <= 1:
                self.best_loss = self.mean_epoch_loss
            else:
                if self.mean_epoch_loss < self.best_loss:
                    self.best_loss = self.mean_epoch_loss

            # print(f'=============== PROFILER ==============\n')
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        if self.params.diagnostic_logs:
            with torch.no_grad():
                diagnostic_logs['train_loss'] = loss 
                if dist.is_initialized():
                    dist.all_reduce(torch.tensor(diagnostic_logs['train_loss']).to(self.device))
                    diagnostic_logs['train_loss'] = float(diagnostic_logs['train_loss'] / dist.get_world_size())
                logs = {'train_loss': diagnostic_logs['train_loss'], 'epoch': self.epoch}
                # if self.params.log_to_wandb:
                #     wandb.log(logs, step=self.iters)
                return tr_time, data_time, diagnostic_logs
        else:
            with torch.no_grad():
                logs = {'train_loss': loss, 'epoch': self.epoch}

                if dist.is_initialized():
                    for key in sorted(logs.keys()):
                        if isinstance(logs[key], (int, float)):
                            logs[key]= torch.tensor(logs[key]).to(self.device)
                        dist.all_reduce(logs[key])
                        logs[key] = float(logs[key] / dist.get_world_size())

            # if self.params.log_to_wandb:``
            #     wandb.log(logs, step=self.iters)


        return tr_time, data_time, logs

    def validate_one_epoch(self):
        self.model.eval()
        n_valid_batches = 50

        valid_start = time.time()

        valid_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.valid_dataloader):
                if i >= n_valid_batches:
                    break

                inputs, labels = data[0].to(self.device, dtype=torch.float32), data[1].to(self.device, dtype=torch.float32)


                if self.params['integrator'] == 'E1':
                    outputs = self.E1_integrator(inputs, train=False)
                elif self.params['integrator'] == 'RK2':
                    outputs = self.RK2_integrator(inputs, train=False)
                elif self.params['integrator'] == 'RK4':
                    outputs = self.RK4_integrator(inputs, train=False)

                else:
                    outputs = self.model(inputs, train=False)

                if dist.is_initialized():
                    loss = self.model.module.forward_loss(labels, outputs)

                    if self.params['spectral_loss']:
                        loss += self.model.module.spectral_loss(labels, outputs, self.params['spectral_loss_weight'], self.params['spectral_loss_threshold_wavenumber'])
                else:
                    loss = self.model.forward_loss(labels, outputs)

                    if self.params['spectral_loss']:
                        loss += self.model.spectral_loss(labels, outputs, self.params['spectral_loss_weight'], self.params['spectral_loss_threshold_wavenumber'])

                # check valid pred
                self.val_pred = outputs

                valid_loss += loss

                if self.params.log_to_wandb:
                    if (self.epoch % self.params.wandb_table_logging_interval == 1) and (i == 0):
                        logging.info("Logging validation [input, target, prediction] to wandb table.")
                        _wandb_table = wandb.Table(columns=self.wandb_table.columns, data=self.wandb_table.data)
                        _wandb_table = log_input_target_prediction(inputs, labels, outputs, _wandb_table, self.iters)
                        wandb.log({f"EPOCH {self.epoch} Validation Input/Target/Prediction" : _wandb_table}, step=self.iters)
                        #self.wandb_table = _wandb_table

        valid_time = time.time() - valid_start

        self.valid_loss = valid_loss/n_valid_batches
        logs = {'valid_loss': valid_loss / n_valid_batches, 'epoch': self.epoch}
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                if isinstance(logs[key], (int, float)):
                    logs[key] = torch.tensor(logs[key]).to(self.device)
                dist.all_reduce(logs[key])
                logs[key] = float(logs[key] / dist.get_world_size())

        # if self.params.log_to_wandb:
        #     wandb.log(logs, self.iters)


        return valid_time, logs


    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model

        checkpoint = {
            'iters': self.iters, 
            'epochs': self.epoch, 
            'best_valid_loss': self.best_valid_loss,
            'model_state': model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        # Save scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)


    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, value in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = value
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epochs']
        self.best_valid_loss = checkpoint.get('best_valid_loss', 1.e6)
        print(f'START EPOCH:', self.startEpoch)

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler state if it exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info('Learning rate scheduler state restored')

    def E1_integrator(self, input, train=False):
        """
        input: B, C, T, H, W
        """
        k1 = self.model(input, train=train)
        
        return input + k1


    def RK2_integrator(self, input, train=False):
        """
        input: B, C, T, H, W
        """
        k1 = self.model(input, train=train)
        k2 = self.model(input + k1, train=train)
    
        return input + 0.5 * (k1 + k2)


    def RK4_integrator(self, input, train=False):
        """
        input: B, C, T, H, W
        Note: https://arxiv.org/abs/2304.07029
        "Long-term instabilities of deep learning-based digital twins of the climate system: The cause and a solution"
        """
        k1 = self.model(input, train=train)
        k2 = self.model(input + 0.5 * k1, train=train) 
        k3 = self.model(input + 0.5 * k2, train=train)
        k4 = self.model(input + k3, train=train)
    
        return input + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
