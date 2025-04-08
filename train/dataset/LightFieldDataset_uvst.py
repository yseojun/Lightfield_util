import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from gen_uvst import get_uvst, get_uvst_time

class LightFieldDataSet(Dataset):
    def __init__(self, args):
        self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
        self.grid_size = args.grid_size
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        self.H, self.W = self.image_load(0).shape[-2:]


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
        
        uvst = get_uvst(x, y, self.H, self.W, 1, 0.1)
        
        sample = {'img': tensor_image, 'img_idx': idx, 'uvst': uvst}
        
        return sample

class LightFieldDataSet_dynamic(Dataset):
    def __init__(self, args):
        self.data_path = args.data_path
        self.grid_size = args.grid_size
        self.num_frames = args.num_frames
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        
        # 모든 뷰 폴더 찾기 (00_00, 00_01 등)
        self.view_dirs = sorted([d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d)) and '_' in d])
        
        sample_img_path = os.path.join(self.data_path, self.view_dirs[0], '000.png')
        sample_img = read_image(sample_img_path)
        self.H, self.W = sample_img.shape[-2:]
        
        self.frame_paths = []
        for view_dir in self.view_dirs:
            view_path = os.path.join(self.data_path, view_dir)
            frames = sorted([f for f in os.listdir(view_path) if f.endswith('.png')])
            for frame in frames:
                self.frame_paths.append(os.path.join(view_path, frame))
                
    def img_load(self, idx):
        img = read_image(self.frame_paths[idx])
        return img / 255.

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        tensor_image = self.img_transform(self.img_load(idx))
        
        path_parts = self.frame_paths[idx].split(os.sep)
        view_dir = path_parts[-2]  # 뷰 디렉토리 이름 (예: "00_01")
        frame_file = path_parts[-1]  # 프레임 파일 이름 (예: "000.png")
        
        y, x = map(int, view_dir.split('_'))
        
        t = int(os.path.splitext(frame_file)[0])
        
        uvst = get_uvst_time(x, y, t, self.H, self.W, self.num_frames, 1, 0.1)
        
        sample = {'img': tensor_image, 'img_idx': idx, 'uvst': uvst}
        
        return sample