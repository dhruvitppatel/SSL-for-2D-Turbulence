import os, re, sys, logging

import torch
import torch.distributed as dist
from ruamel.yaml import YAML
import yaml
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(parent_dir)
sys.path.append(src_dir)

from src.models.vision_transformer import ViT
from src.utils.data_loaders import get_dataloader

from analysis.short_analysis import perform_short_analysis
from analysis.long_analysis_parallel import perform_long_analysis
from analysis.io_utils import load_params, get_npy_files
from analysis.rollout import n_step_rollout, single_step_rollout

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()

def main(config):
    """
    Main function to run analysis on trained model.
    """
    # Extract values from config
    dataset_params = config["dataset_params"]
    short_analysis_params = config["short_analysis_params"]
    long_analysis_params = config["long_analysis_params"]

    # Dataset-related parameters
    run_num = dataset_params["run_num"]
    root_dir = os.path.join(dataset_params["root_dir"], run_num)
    model_filename = dataset_params["model_filename"]
    params_filename = dataset_params["params_filename"]

    # Short analysis parameters
    rmse = short_analysis_params["rmse"]
    acc = short_analysis_params["acc"]
    spectra = short_analysis_params["spectra"]
    spectra_leadtimes = short_analysis_params["spectra_leadtimes"]
    test_length_short = short_analysis_params["analysis_length"]
    num_tests = short_analysis_params["num_ensembles"]
    test_file_start_idx = short_analysis_params["test_file_start_idx"]
    test_length_climo = short_analysis_params["test_length_climo"]

    # Long analysis parameters
    test_length_long = long_analysis_params["analysis_length"]
    save_data_length = long_analysis_params["save_data_length"]
    zonal_eof_pc = long_analysis_params["zonal_eof_pc"]
    eof_ncomp = long_analysis_params["eof_ncomp"]
    video = long_analysis_params["video"]
    video_length = long_analysis_params["video_length"]
    div = long_analysis_params["div"]
    return_period = long_analysis_params["return_period"]
    temporal_mean = long_analysis_params["temporal_mean"]
    zonal_mean = long_analysis_params["zonal_mean"]

    # Read in params file
    params_fp = os.path.join(root_dir, params_filename)
    train_params = load_params(params_fp)
    logging.info(f'params: {train_params}')

    # Initiate model and load weights
    model_fp = os.path.join(root_dir, model_filename)
    model = ViT(
        img_size=train_params["img_size"],
        patch_size=train_params["patch_size"],
        num_frames=train_params["num_frames"],
        tubelet_size=train_params["tubelet_size"],
        in_chans=train_params["in_chans"],
        encoder_embed_dim=train_params["encoder_embed_dim"],
        encoder_depth=train_params["encoder_depth"],
        encoder_num_heads=train_params["encoder_num_heads"],
        decoder_embed_dim=train_params["decoder_embed_dim"],
        decoder_depth=train_params["decoder_depth"],
        decoder_num_heads=train_params["decoder_num_heads"],
        mlp_ratio=train_params["mlp_ratio"],
        num_out_frames=train_params["num_out_frames"],
        patch_recovery=train_params["patch_recovery"]
        )

    ckpt_temp = torch.load(model_fp, map_location=torch.device(device), weights_only=False)['model_state']

    # Training single vs multiple gpu
    #if gpu_count > 1:
    ckpt = {key[7:]: val for key, val in ckpt_temp.items()}
    #else:
    #    ckpt = ckpt_temp
    model.load_state_dict(ckpt)
    model.eval()

    print(f'TORCH DEVICE: {device}')
    model.to(device)
    print(f'MODEL.device: {next(model.parameters()).device}')

    # Directory to saved emulated data and analysis
    save_dir = os.path.join(root_dir, 'data')  # or create a separate directory if desired
    analysis_dir = os.path.join(root_dir, 'analysis')
    plot_dir = os.path.join(root_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Initiate dataloaders
    test_file_range = (test_file_start_idx, test_file_start_idx+(test_length_short*(num_tests+2)*train_params["target_step"])-1)
    dataloader, dataset = get_dataloader(data_dir=train_params["data_dir"],
                                        file_range=test_file_range,
                                        target_step=train_params["target_step"],
                                        train_tendencies=train_params["train_tendencies"],
                                        batch_size=test_length_short,
                                        train=False,
                                        stride=train_params["target_step"],
                                        distributed=dist.is_initialized(),
                                        num_frames=train_params["num_frames"],
                                        num_out_frames=train_params["num_out_frames"],
                                        num_workers=2,
                                        pin_memory=train_params["pin_memory"],
                                        forward_step=train_params["forward_step"])

    # Perform short analysis
    if rmse or acc or spectra:

        # Dataloader to calculate climatology
        test_file_range = (test_file_start_idx, test_file_start_idx+test_length_climo)
        dataloader_climo, dataset_climo = get_dataloader(data_dir=train_params["data_dir"],
                                        file_range=test_file_range,
                                        target_step=1,
                                        train_tendencies=train_params["train_tendencies"],
                                        batch_size=test_length_climo,
                                        train=False,
                                        stride=1,
                                        distributed=dist.is_initialized(),
                                        num_frames=1, #params["num_frames"],
                                        num_out_frames=1, #params["num_out_frames"],
                                        num_workers=2,
                                        pin_memory=train_params["pin_memory"],
                                        forward_step=train_params["forward_step"])

        climo_data, _ = next(iter(dataloader_climo))
        print(f'climo_data.shape: {climo_data.shape}')

        climo_data = climo_data.transpose(-1, -2).squeeze().detach().cpu().numpy()                     # [B=n_steps, C, X, Y]
        climo_u = climo_data[:,0].mean(axis=0)                                       # [X, Y]
        climo_v = climo_data[:,1].mean(axis=0) 

        results_short = perform_short_analysis(model, dataloader, dataset, climo_u, climo_v, short_analysis_params, train_params, dataset_params, device)
        print('short analysis performed')

        if short_analysis_params["save_short_analysis"]:
            print('saving short analysis results')
            save_fp = os.path.join(analysis_dir, 'emulate')
            os.makedirs(save_fp, exist_ok=True)
            np.savez(os.path.join(save_fp, 'short_analysis.npz'), **results_short)

    else:
        print('No short analysis requested')



    if long_analysis_params["long_analysis_emulator"]:

        # Number of files to be saved
        save_data_length = np.maximum(long_analysis_params["save_data_length"], long_analysis_params["analysis_length"])

        # Check if data exists in the saved folder
        files = get_npy_files(save_dir)

        rollout_length = save_data_length

        C = train_params["in_chans"]
        T = train_params["num_frames"]
        H = train_params["img_size"]
        W = train_params["img_size"]
        chunk_size = long_analysis_params["num_snapshots_per_file"]

        if T > chunk_size:
            raise ValueError(f"num_frames ({T}) exceeds num_snapshots_per_file ({chunk_size}).")

        if files:

            last_file = files[-1]

            try:
                chunk = np.load(os.path.join(save_dir, last_file))  # expects shape (chunk_size, C, H, W)
            except Exception:
                # drop corrupted final file and retry once
                print(Exception)
                files.pop()
                last_file = files[-1]
                chunk = np.load(os.path.join(save_dir, last_file))

            if chunk.ndim != 4:
                raise ValueError(f"Expected 4D chunk but got shape {chunk.shape}")

            # grab the last T frames
            last_T = chunk[-T:][::-1]  # shape (T, C, H, W) # Note temporal order is: (t, t-1, t-2, ...)

            print('Loading saved data')
            # normalize & stack into data[c, t, h, w]
            data = np.empty((C, T, H, W), dtype=np.float32) # shape (C, T, H, W)   # Note temporal order of ic: (t, t-1, t-2, ...)

            for t, frame in enumerate(last_T):
                for ch in range(C):
                    data[ch, t] = (frame[ch] - dataset.input_mean[ch]) / dataset.input_std[ch]


            print(f'Loaded npy IC shape (c, t, h, w) {data.shape}')
            print(f'Resuming emulation from file {files[-1]}')
            ic = torch.tensor(data, dtype=torch.float32).unsqueeze(dim=0).transpose(-1, -2).to(device)  # Convert to tensor with correct dtype
            
            # extract numeric iteration from filename (e.g. "20" from "20.npy")
            idx = int(re.findall(r"\d+", last_file)[-1])
            current_rollout_length = idx
            print(f"Loaded IC from {last_file}, resuming at step {idx}.")
        else:

            # initializing dataloader for loading intial conditions
            test_file_range_video = (test_file_start_idx, test_file_start_idx+(video_length*train_params["target_step"])+train_params["target_step"])
            dataloader_video, dataset_video = get_dataloader(data_dir=train_params["data_dir"],
                                                            file_range=test_file_range_video,
                                                            target_step=train_params["target_step"],
                                                            train_tendencies=train_params["train_tendencies"],
                                                            batch_size=video_length,
                                                            train=False,
                                                            stride=train_params["target_step"],
                                                            distributed=dist.is_initialized(),
                                                            num_frames=train_params["num_frames"],
                                                            num_out_frames=train_params["num_out_frames"],
                                                            num_workers=2,
                                                            pin_memory=train_params["pin_memory"],
                                                            forward_step=train_params["forward_step"])

            inp, tar = next(iter(dataloader_video))

            # no prior files
            ic = inp[0].unsqueeze(0).to(device, dtype=torch.float32)
            current_rollout_length = 0
            print("No existing files—starting fresh.")

        print("IC shape:", ic.shape, "| resume step:", current_rollout_length, " | 'IC.device: ",ic.device)

        pred_np_arr = []
        for i in range(current_rollout_length, rollout_length):
            pred, ic = single_step_rollout(model, ic, train_tendencies=train_params["train_tendencies"])

            # if (i+1)%1000==0:
            #     print(f'#{i+1} ic shape {ic.shape} -- Pred {pred.shape} ')

            pred_np = pred.clone().transpose(-1,-2).squeeze().detach().cpu().numpy()

            # print(f'pred_np {pred_np.shape} -- pred_np[0,:] {pred_np[0,:].shape}', )

            # Unnormalize data
            for ch in range(C):
                pred_np[ch] = (pred_np[ch] * dataset.input_std[ch]) + dataset.input_mean[ch]

            # Saving chunks of data
            pred_np_arr.append(pred_np)

            if (i+1) % chunk_size == 0:
                np.save(save_dir + f'/{i+1}.npy', np.stack(pred_np_arr, axis=0))
                print(f'{i+1}.npy saved', np.asarray(pred_np_arr).shape)

                pred_np_arr = []


    if long_analysis_params["temporal_mean"] or long_analysis_params["zonal_mean"] or long_analysis_params["spectra"] or \
        long_analysis_params["zonal_U"] or long_analysis_params["zonal_V"] or long_analysis_params["zonal_Omega"] or \
        long_analysis_params["zonal_eof_pc"] or long_analysis_params["div"] or long_analysis_params["video"] or \
        long_analysis_params["return_period"] or long_analysis_params["return_period_anomaly"] or long_analysis_params["PDF_U"] or long_analysis_params["PDF_Omega"]:
        perform_long_analysis(save_dir, analysis_dir, dataset_params, long_analysis_params, train_params)
        print('long analysis performed')
        print(f'save_dir: {save_dir}')
        print(f'analysis_dir: {analysis_dir}')

    else:
        print('No long analysis requested')

    # # Combine results
    # results = {**results_short, **results_long}

    # # Plot analysis
    # plot_analysis(results, analysis_dict, run_num, save_dir)

    return 



if __name__ == "__main__":
    # Load the configuration file

    config_filename = str(sys.argv[1])
    with open("config/" + config_filename, "r") as f:
        config = yaml.safe_load(f)

    # Pass the entire config dictionary to main
    main(config)
