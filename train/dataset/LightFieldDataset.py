import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class LightFieldDataSet(Dataset):
    def __init__(self, args):
        self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
        self.grid_size = args.grid_size
        self.crop_list, self.resize_list = args.crop_list, args.resize_list


    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1,0,1)
        return img / 255.

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        tensor_image = self.img_transform(self.img_load(idx))
        
        y = idx // self.grid_size
        x = idx % self.grid_size
        
        y = float((2.0 * (y / (self.grid_size - 1))) - 1.0)
        x = -float((2.0 * (x / (self.grid_size - 1))) - 1.0)
        
        sample = {'img': tensor_image, 'img_idx': idx, 'norm_y': y, 'norm_x': x}
        
        return sample