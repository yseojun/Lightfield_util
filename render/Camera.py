import numpy as np

class Camera:
    def __init__(self, x=0, y=0, z=0, theta=0, phi=0, fov=90, H=256, W=512):
        self.H = H
        self.W = W
        self.K = self.calculate_intrinsic()

        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.phi = phi

        self.fov = fov

        self.c2w = self.set_c2w()

        self.uv_depth = 2.4
        self.st_depth = 0
        
        self.plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]), (H * W, 3))
        self.plane_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),(self.H * self.W, 3))
        self.plane_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),(self.H * self.W, 3))
                
    def calculate_intrinsic(self):
        cx = self.W / 2
        cy = self.H / 2

        focal_length = np.sqrt(self.W**2 + self.H**2)

        intrinsic = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        return intrinsic

    def get_uvst_one_plane(self):
        aspect = self.W/self.H
        uv_scale = 1.0
        st_scale = 0.1 
        
        u = np.linspace(-1, 1, self.W, dtype='float32')
        v = np.linspace(1, -1, self.H, dtype='float32') / aspect
        vu = list(np.meshgrid(u, v))

        u = vu[0] * uv_scale
        v = vu[1] * uv_scale

        x, y, z = self.c2w[0, 3], self.c2w[1, 3], self.c2w[2, 3]
        
        f = 5/14 
        
        x_min, x_max = -z * f, z * f
        y_min, y_max = z * f, -z * f
        
        s = np.linspace(x_min, x_max, self.W, dtype='float32') + x
        t = np.linspace(y_min, y_max, self.H, dtype='float32') / aspect + y
        st = list(np.meshgrid(s, t))
        
        s = st[0] * st_scale
        t = st[1] * st_scale

        uvst = np.stack((u, v, s, t), axis=-1)
        
        return uvst
    
    def get_uvst(self):
        rays_o, rays_d = self.get_rays_o_d()

        inter_uv = self.get_rays_inter_with_plane(self.plane_uv, rays_o, rays_d)
        inter_st = self.get_rays_inter_with_plane(self.plane_st, rays_o, rays_d)
        uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)

        return uvst
    
    def get_rays_o_d(self):
        i, j = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-self.K[0][2])/self.K[0][0], -(j-self.K[1][2])/self.K[1][1], -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., np.newaxis, :] * self.c2w[:3,:3], -1)
        rays_o = np.broadcast_to(self.c2w[:3,-1], np.shape(rays_d))

        rays_o = np.reshape(rays_o,(-1,3))
        rays_d = np.reshape(rays_d,(-1,3))
        
        return rays_o, rays_d
    
    def get_rays_inter_with_plane(self, p0, rays_o, rays_d):
        s1 = np.sum(p0 * self.plane_normal,1)

        s2 = np.sum(rays_o * self.plane_normal,1)
        s3 = np.sum(rays_d * self.plane_normal,1) 

        dist = (s1 - s2) / s3
        dist_group = np.broadcast_to(dist,(3, dist.shape[0])).T
        inter_point = rays_o + dist_group * rays_d

        return inter_point
    
    def get_camera_translation(self):
        current_T = self.c2w[:3,-1].T
        current_T = np.expand_dims(current_T,axis=0)
        return current_T
    
    def set_c2w(self):
        theta = np.radians(self.theta)
        phi = np.radians(self.phi)
        psi = np.pi
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        R = R_z @ R_y @ R_x
        
        T = np.array([self.x, self.y, self.z])
        
        c2w = np.zeros((3, 4))
        c2w[:3, :3] = R
        c2w[:3, 3] = T

        self.c2w = c2w
        return c2w

