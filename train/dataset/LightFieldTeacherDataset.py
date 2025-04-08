import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class LightFieldTeacherDataSet(Dataset):
    def __init__(self, main_dir, transform, train=True):
        self.main_dir = main_dir
        self.transform = transform
        self.img_list = [os.path.join(main_dir, img_id) for img_id in os.listdir(main_dir)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        image = Image.open(img_name).convert("RGB")
        
        coords = os.path.splitext(os.path.basename(img_name))[0].split('_')
        x = torch.tensor(float(coords[0]), dtype=torch.float32)
        y = torch.tensor(float(coords[1]), dtype=torch.float32)
        
        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)

        data_dict = {
            "norm_x": x,
            "norm_y": y,
            "img_gt": tensor_image,
        }
        return data_dict