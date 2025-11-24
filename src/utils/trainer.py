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
#from torch.profiler import profile, record_function, ProfilerActivity


#torch.backends.cuda.enable_flash_sdp(True)

class Trainer():
    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.early_stop_epoch = params['early_stop_epoch'] - 1 if 'early_stop_epoch' in params else None
        

        logging.info('rank %d, begin data loader init' % world_rank)
        print(params)


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

 
        if params.log_to_wandb:
            wandb.init(config=params, name=params.name, group=params.group, project=params.project)

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
       

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            checkpoint_path = natsorted([file for file in glob.glob(self.params.checkpoint_path_globstr) if os.path.isfile(file)])[-1]
            print(f'RESTORE CKPT: {checkpoint_path}')
            self.restore_checkpoint(checkpoint_path)
        else:
            logging.info("Starting fresh training run")

        self.epoch = self.startEpoch


        # Set learning rate scheduluer
        if params["scheduler"] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=float(params['factor']), patience=int(params['patience']), 
                                                                        cooldown=int(params['cooldown']), mode='min')
        elif params["scheduler"] == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params["max_epochs"], eta_min=float(params['lr_min']),
                                                                        last_epoch=self.startEpoch-1)
        elif params["scheduler"] == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=int(params["T_0"]), 
                                                                                  T_mult=int(params["T_mult"]), eta_min=float(params['lr_min']),
                                                                                  last_epoch=self.startEpoch-1)
        else:
            self.scheduler = None

        # Warm up epochs if using
        if params.warmup:
            self.warmuplr = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=params.warmup_startfactor,
                                                              total_iters=params.warmup_totaliters)


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

        best_valid_loss = 1.e6
        early_stopping_counter = 0
        early_stop_epoch_triggered = False

        for epoch in range(self.params["max_epochs"]):
            
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
            if self.params["warmup"] and (self.startEpoch + epoch) < self.params["warmup_totaliters"]:
                self.warmuplr.step()
            else:
                if self.params["scheduler"] == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_logs['valid_loss'])
                elif self.params["scheduler"] == 'CosineAnnealingLR' or self.params["scheduler"] == 'CosineAnnealingWarmRestarts':
                    self.scheduler.step()
                    if self.epoch >= self.params.max_epochs:
                        logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                        break

            
            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr, 'epoch': self.epoch})


            # Early stopping logic should be outside of world_rank check
            if valid_logs["valid_loss"] <= best_valid_loss:
               best_valid_loss  = valid_logs['valid_loss']
               early_stopping_counter = 0
            else:
               early_stopping_counter += 1


            if self.world_rank == 0:
               if self.params.save_checkpoint:
                  checkpoint_path_out = '_'.join(self.params.checkpoint_path_globstr.split('_')[:-1])
                  self.save_checkpoint(checkpoint_path_out + f'.tar')
                  if valid_logs["valid_loss"] <= best_valid_loss:
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


        ## After training loop ends
        #if self.params.log_to_wandb:
        #   if self.world_rank == 0:
        #      self.log_all_plots_wandb()
        
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

        if self.params.diagnostic_logs:
            diagnostic_logs = {}


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

            loss.backward()

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
                    if self.params.log_to_wandb:
                        wandb.log(diagnostic_logs, step=(self.epoch - 1) * total_iterations + self.iters)

            
            torch.cuda.empty_cache()

            # print(f'=============== PROFILER ==============\n')
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        if self.params.diagnostic_logs:
            with torch.no_grad():
                diagnostic_logs['train_loss'] = loss 
                if dist.is_initialized():
                    dist.all_reduce(torch.tensor(diagnostic_logs['train_loss']).to(self.device))
                    diagnostic_logs['train_loss'] = float(diagnostic_logs['train_loss'] / dist.get_world_size())
                logs = {'train_loss': diagnostic_logs['train_loss'], 'epoch': self.epoch}
                if self.params.log_to_wandb:
                    wandb.log(logs)
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

            if self.params.log_to_wandb:
                wandb.log(logs)


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
                        _wandb_table = log_input_target_prediction(inputs, labels, outputs, _wandb_table, self.epoch)
                        wandb.log({f"EPOCH {self.epoch} Validation Input/Target/Prediction" : _wandb_table})
                        #self.wandb_table = _wandb_table

        valid_time = time.time() - valid_start

        logs = {'valid_loss': valid_loss / n_valid_batches, 'epoch': self.epoch}
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                if isinstance(logs[key], (int, float)):
                    logs[key] = torch.tensor(logs[key]).to(self.device)
                dist.all_reduce(logs[key])
                logs[key] = float(logs[key] / dist.get_world_size())

        if self.params.log_to_wandb:
            wandb.log(logs)


        return valid_time, logs


    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epochs': self.epoch, 'model_state': model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)


    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
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
        print(f'START EPOCH:', self.startEpoch)
        if self.params.resuming:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
