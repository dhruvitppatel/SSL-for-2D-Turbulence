import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Normalize
from scipy.io import loadmat
import numpy as np
from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV


def get_dataloader(data_dir, file_range, target_step, train_tendencies, batch_size, train, distributed, stride=1, 
                   num_frames=1, num_out_frames=1, target_step_hist=None, num_workers=1, pin_memory=True):

    if isinstance(data_dir, list) and len(data_dir)>1:
        assert len(file_range) == len(data_dir), 'len(file_range) should be the same as len(data_dir)'
        dataset = TurbulenceMultiDataset(data_dir=data_dir, file_range=file_range, target_step=target_step,
                                         train_tendencies=train_tendencies, stride=stride, num_frames=num_frames, 
                                         num_out_frames=num_out_frames, target_step_hist=target_step_hist)
    else:
        dataset = TurbulenceDataset(data_dir=data_dir, file_range=file_range, target_step=target_step, 
                                    train_tendencies=train_tendencies, stride=stride, num_frames=num_frames, 
                                    num_out_frames=num_out_frames, target_step_hist=target_step_hist)

    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    if train and not distributed:
        sampler = torch.utils.data.RandomSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,  # (sampler is None)
                                             sampler=sampler,  # if train else None
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)


    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class TurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_range, target_step, train_tendencies, stride, num_frames, num_out_frames, target_step_hist):
        """
        Args:
            data_dir (str): Directory with .mat data files.
            file_range (tuple): Range of file numbers (start, end).
            target_step (int): Number of steps forward for target output.
            train_tendencies (bool): whether to train to tendencies (True) or not (False).
            stride (int): Number of steps between samples.
            num_frames (int): Number of time samples for input.
            num_out_frames (int): Number of time samples for output.
            target_step_hist (int or None): Number of steps for previous samples in input.
        """
        self.data_dir = data_dir
        if isinstance(file_range[0], list):
            temp_inp, temp_label = [], []
            for part in file_range:
                temp_inp.append(list(range(part[0] + (num_frames - 1), part[1] + 1, stride)))
                temp_label.append(list(range(part[0] + (num_frames - 1) + target_step, part[1]+1+target_step, stride)))
            self.input_file_numbers = [item for sublist in temp_inp for item in sublist]
            self.label_file_numbers = [item for sublist in temp_label for item in sublist]
        else:
            self.input_file_numbers = list(range(file_range[0] + (num_frames - 1), file_range[1] + 1, stride))
            self.label_file_numbers = list(range(file_range[0] + (num_frames - 1) + target_step, file_range[1]+1+target_step, stride))
        #self.input_file_list = [os.path.join(data_dir, 'data', f"{i}.mat") for i in self.input_file_numbers]
        #self.label_file_list = [os.path.join(data_dir, 'data', f"{i}.mat") for i in self.label_file_numbers]
        self.target_step = target_step
        self.train_tendencies = train_tendencies
        self.stride = stride
        self.num_frames = num_frames
        self.num_out_frames = num_out_frames
        if target_step_hist is None:
            self.target_step_hist = target_step
        else:
            self.target_step_hist = target_step_hist

        input_mean, input_std = self._get_file_stats(inp=True)
        self.normalize_input = Normalize(input_mean, input_std)

        self.input_mean = input_mean
        self.input_std = input_std
        self.label_mean = input_mean
        self.label_std = input_std

        if self.train_tendencies:
            label_mean, label_std = self._get_file_stats(inp=False)
            self.normalize_label = Normalize(label_mean, label_std)

            self.label_mean = label_mean
            self.label_std = label_std
        else:
            self.normalize_label = self.normalize_input

    
    def _get_file_stats(self, inp):
        """
        Returns:
            mean, std (np.array): [number of channels = 2,] mean and std 
        """
        if inp:
            mean_fp = os.path.join(self.data_dir, 'stats', 'mean_full_field.npy')
            std_fp = os.path.join(self.data_dir, 'stats', 'std_full_field.npy')
            #mean_std_data = loadmat(os.path.join(self.data_dir, 'stats', 'mean_std_DNS_NX64_dt0.0005_IC1.mat_1.0.mat'))
            mean_std_data = loadmat(os.path.join(self.data_dir, 'stats', 'mean_std_DNS_NX256_dt0.0002_IC1.mat_1.0.mat'))
        else:
            mean_fp = os.path.join(self.data_dir, 'stats', 'mean_tendencies.npy')
            std_fp = os.path.join(self.data_dir, 'stats', 'std_tendencies.npy')

        mean = np.asarray([mean_std_data['U_mean'], mean_std_data['V_mean']])
        std = np.asarray([mean_std_data['U_std'], mean_std_data['V_std'] ])
        # mean = list(np.load(mean_fp)) 
        # std = list(np.load(std_fp))
        print(f'mean: {mean}')
        print(f'std: {std}')

        return mean, std

    def __len__(self):
        return len(self.input_file_numbers)

    def __getitem__(self, idx):
        """
        Args:
          idx (int): Index of the file to load.

        Returns:
          tuple (torch.Tensor): Data loaded from the .mat file.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inp_tensor, label_tensor = [], []
        for t in range(self.num_frames):
            file_num = self.input_file_numbers[idx] - t*self.target_step_hist
            inp_data = self.get_item_input_one_step(file_num)   # output (tensor) shape: [C=2, T=1, X, Y]
            inp_tensor.append(inp_data)
        for t in range(self.num_out_frames):
            file_num = self.label_file_numbers[idx] - t*self.target_step_hist
            label_data = self.get_item_label_one_step(file_num)   # output (tensor): [C=2, T=1, X, Y]
            label_tensor.append(label_data)

        inp_tensor = torch.cat(inp_tensor, dim=1)  # along T dim
        label_tensor = torch.cat(label_tensor, dim=1)

        return (inp_tensor, label_tensor)

    def get_item_input_one_step(self, idx):

        input_file_path = os.path.join(self.data_dir, 'data', f"{idx}.mat")
        input_mat_data = loadmat(input_file_path)
        input_Omega = input_mat_data['Omega']
        input_data_tensor = self.omega2uv(input_Omega) #.unsqueeze(1)
        input_data_tensor = self.normalize_input(input_data_tensor).unsqueeze(1)

        return input_data_tensor

    def get_item_label_one_step(self, idx):

        label_file_path = os.path.join(self.data_dir, 'data', f"{idx}.mat")
        label_mat_data = loadmat(label_file_path)
        label_Omega = label_mat_data['Omega']
        label_data_tensor = self.omega2uv(label_Omega) #.unsqueeze(1)

        if self.train_tendencies:
            # Fetch previous time step and subtract it from label
            input_file_path = os.path.join(self.data_dir, 'data', f"{idx - self.target_step}.mat")
            input_mat_data = loadmat(input_file_path)
            input_Omega = input_mat_data['Omega']
            input_data_tensor = self.omega2uv(input_Omega)
            label_data_tensor -= input_data_tensor


        label_data_tensor = self.normalize_label(label_data_tensor).unsqueeze(1)

        return label_data_tensor

    def omega2uv(self, Omega):
        """
        Args:
          Omega (np.array): 2D Omega data.
        Returns:
          data_tensor (torch.Tensor): U, V data.
        """
        nx, ny = Omega.shape
        Lx, Ly = 2 * np.pi, 2 * np.pi
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij')

        Psi = Omega2Psi(Omega, invKsq)
        U, V = Psi2UV(Psi, Kx, Ky)

        # Combine U and V into a single tensor with 2 channels
        data_tensor = torch.tensor(np.stack([U, V]), dtype=torch.float32)

        return data_tensor


class TurbulenceMultiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_range, target_step, train_tendencies, stride, num_frames, num_out_frames, target_step_hist):

        self.datasets = []
        self.dlens = []
        for i in range(data_dir):
            dataset = TurbulenceDataset(data_dir[i], file_range[i], target_step, train_tendencies, stride, num_frames,
                                         num_out_frames, target_step_hist)
            self.datasets.append(dataset)
            self.dlens.append(len(dataset))

    def __len__(self):
        return sum(self.dlens)

    def __getitem__(self, idx):

        cum_ids = np.cumsum(self.dlens)
        a = (cum_ids - 1) >= idx
        dset_idx = np.argwhere(a)[0]
        sample_idx = idx - np.concatenate(([0], cum_ids))[dset_idx]

        return self.datasets[dset_idx][sample_idx]
