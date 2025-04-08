import numpy as np

def get_uvst(x, y, H, W, uv_scale=1, st_scale=0.1):
    aspect = W / H
    u = np.linspace(-1, 1, W, dtype=np.float32)
    v = np.linspace(-1, 1, H, dtype=np.float32) / aspect
    vu = np.meshgrid(u, v)
    
    u = vu[0] * uv_scale
    v = vu[1] * uv_scale
    
    s = np.ones_like(vu[0]) * x * st_scale
    t = np.ones_like(vu[1]) * y * st_scale
    uvst = np.stack([u, v, s, t], axis=-1)
    
    return uvst

def get_uvst_time(x, y, tt, H, W, num_frames=50, uv_scale=1, st_scale=0.1):
    aspect = W / H
    u = np.linspace(-1, 1, W, dtype=np.float32)
    v = np.linspace(-1, 1, H, dtype=np.float32) / aspect
    vu = np.meshgrid(u, v)
    
    u = vu[0] * uv_scale
    v = vu[1] * uv_scale
    
    s = np.ones_like(vu[0]) * x * st_scale
    t = np.ones_like(vu[1]) * y * st_scale
    
    time = tt / (num_frames - 1)
    time = np.ones_like(vu[0]) * time
    uvst = np.stack([u, v, s, t, time], axis=-1)
    
    return uvst

def __main__():
    # test
    uvst = get_uvst(0, 0, 1, 0.1, 1024, 1024)
    print(uvst.shape)
    print(uvst)
    uvst_time = get_uvst_time(0, 0, 3, 1, 0.1, 1024, 1024, 10)
    print(uvst_time.shape)
    print(uvst_time)

if __name__ == "__main__":
    __main__()