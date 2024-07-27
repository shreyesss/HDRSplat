import numpy as np
import os 
import cv2
import trimesh
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, focal2fov, getIntrinsicMatrix

def project_point_cloud(save_dir,mesh_path):

    def load_camera(scene_path):
        znear = 0.1
        zfar = 1000
    
        cameras_extrinsic_file = os.path.join(scene_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(scene_path, "sparse/0", "cameras.bin")
    
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    

        key = 10
        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        t = np.array(extr.tvec).reshape(3,1)
        K = getIntrinsicMatrix(intr.params)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

        return R, t, K,height,width, extr.name, extr.xys 

    R, t, K, h, w, image_name, xys = load_camera(scene_path)
   

    def get_2D_points(mesh,K,T,h,w):

        Pw = mesh.vertices # [N,3]
        Pw = np.hstack((Pw, np.ones((len(Pw), 1)))).T # [4,N]
        Pc = np.dot(T, Pw).T # [N,3]
        Pc_norm = (Pc / Pc[:, 2][:, np.newaxis]).T #[N,3]
        uv = np.dot(K, Pc_norm).T
        print(uv, uv.min(), uv.max())
     
        uv_new = uv[(uv[:, 0] >= 0) & (uv[:, 0] <= w) &
                                    (uv[:, 1] >= 0) & (uv[:, 1] <= h)]
        return uv_new[:,0].astype(np.uint32), uv_new[:,1].astype(np.uint32)

  
 
   
   
    mesh = trimesh.load_mesh(mesh_path)
    image_path = os.path.join(scene_path, "images", image_name)

    image = cv2.imread(image_path)
   
   
    T = getWorld2View2(R,t[:,0])[:3,:]
    X, Y = get_2D_points(mesh1,K,T,h,w)
    # x2, y2 = get_2D_points(mesh2,K,T)

    for (x,y) in zip(X,Y):
        cv2.circle(image1, (x,y), 2, (255, 255, 0), -1)
    cv2.imwrite(os.path.join(save_dir, "pointcloud-projected.png"), image1)

 

save_dir = "./"
mesh_path = "/home/cilab/shreyas/codes/gaussian-splatting/our-dataset/hostelroom-LDR/input.ply"
project_point_cloud(save_dir,mesh_path)

