"""
    This is not library quality code
"""
import shutil

import trimesh
import xml.etree.ElementTree as ET
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from einops import rearrange
from scipy.spatial import ConvexHull
import os

lambda_0 = 0 # 3e-2

def get_info(mesh_location):
    mesh = trimesh.load(mesh_location, force="mesh")
    if not mesh.is_watertight:
        mesh.fill_holes()
    return mesh.center_mass, mesh.volume

def get_mesh_info(model_xml_location):
    tree = ET.parse(model_xml_location)
    root = tree.getroot()
    mesh_assets = {}
    for m in root.findall("./asset/mesh"):
        name, fname = m.get("name"), m.get("file")
        if name and fname:
            mesh_assets[name] = fname
    return mesh_assets

# 1) build the cvxpy problem once for n geoms:
def make_mass_dist_layer(n_geoms):
    # decision variable: densities d ∈ ℝⁿ
    d = cp.Variable(n_geoms)

    # parameters:
    M = cp.Parameter((3, n_geoms))   # p_i * v_i packed into a 3×n matrix
    v = cp.Parameter(n_geoms)        # volumes v_i
    T = cp.Parameter()               # total_mass = Σ_i v_i d_i(orig)
    p = cp.Parameter(3)              # total mass scaled target center of mass
    
    # center‑of‑mass formula
    un_avg_com = M @ d
    dist = un_avg_com - p
    regularization = lambda_0 * cp.norm(d, 2)  # L1 norm regularization
    objective = cp.Minimize(cp.norm(dist) + regularization)

    constraints = [v @ d == T,
                   d >= 0]

    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[M, v, T, p], variables=[d])


# 2) for a batch of XMLs + target_COMs, extract parameters:
def extract_batch_params(xml_paths, target_coms):
    batch_M, batch_v, batch_T, batch_t = [], [], [], []
    batch_roots, batch_geoms = [], []

    for xml_path, target in zip(xml_paths, target_coms):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        mesh_assets = get_mesh_info(xml_path)
        geoms = root.findall(".//geom")

        # collect per‑geom info
        positions, volumes, densities = [], [], []
        dirname = os.path.dirname(xml_path)
        for g in geoms:
            p, vol = get_info(os.path.join(dirname, mesh_assets[g.get("mesh")]))
            positions.append(p)
            volumes.append(vol)
            densities.append(float(g.get("density")))
        
        volumes = np.array(volumes)                  # shape (n,)
        positions = np.stack(positions, axis=1)      # shape (3, n)
        densities = np.array(densities)

        # M_ij = v_j * p_{i,j}
        M_np = positions * volumes[None, :]          # (3, n)
        total_mass = (volumes * densities).sum()     # scalar

        # stash
        batch_M.append(torch.from_numpy(M_np).float())
        batch_v.append(torch.from_numpy(volumes).float())
        batch_T.append(torch.tensor(total_mass).float())
        batch_t.append(torch.from_numpy(target).float())
        batch_roots.append(root)
        batch_geoms.append(geoms)

    # stack into PyTorch tensors with batch dim B
    M_b   = torch.stack(batch_M)   # (B, 3, n)
    v_b   = torch.stack(batch_v)   # (B,   n)
    T_b   = torch.stack(batch_T)   # (B,)
    t_b   = torch.stack(batch_t)   # (B, 3)
    
    p_b = T_b[:, None] * t_b

    return M_b, v_b, T_b, p_b, batch_roots, batch_geoms


# 3) solve the whole batch in one forward pass:
def solve_and_update(xml_paths, target_coms):
    # extract
    M_b, v_b, T_b, p_b, roots, geoms_batch = extract_batch_params(xml_paths, target_coms)
    B, _, n = M_b.shape

    # create (or cache!) one layer for this n
    layer = make_mass_dist_layer(n)

    # forward solve: yields d_opt of shape (B, n)
    d_opt, = layer(M_b, v_b, T_b, p_b, solver_args={'max_iters': 10000})
    
    un_avg_com = M_b @ rearrange(d_opt, "batch n_geoms -> batch n_geoms 1")

    residual = un_avg_com.squeeze() - p_b
    avg_residual = residual / T_b[:, None]

    avg_distance = torch.mean(torch.linalg.norm(avg_residual, dim=1))

    print("Average distance from target COM:", avg_distance.item())


    # write back into each XML
    for root, geoms, d_vec, xml_path in zip(roots, geoms_batch, d_opt.detach().cpu().numpy(), xml_paths):
        for g, new_d in zip(geoms, d_vec):
            g.set("density", str(max(new_d, 0.00005)))
        ET.ElementTree(root).write(xml_path)

class VecMJCFHandler:
    def __init__(self, object_location, random_vec):
        self.object_location = object_location
        self.random_vec = random_vec

        (
                self.friction, 
                self.density, 
                self.scale, 
                self.center_of_mass_x,
                self.center_of_mass_y,
                self.center_of_mass_z
        ) = self.random_vec.T

        self.center_of_mass_x = self.center_of_mass_x.reshape(-1, 1)
        self.center_of_mass_y = self.center_of_mass_y.reshape(-1, 1)
        self.center_of_mass_z = self.center_of_mass_z.reshape(-1, 1)
        
        self.center_of_mass = np.hstack([self.center_of_mass_x,
                                         self.center_of_mass_y,
                                         self.center_of_mass_z])
        
        self.created_files = []
        self.num_environments = self.random_vec.shape[0]
        self.object_dir_name = os.path.dirname(self.object_location)

    def __enter__(self):
        for i in range(self.num_environments):
            current_file_path = self.object_dir_name + f"/model_{i}.xml"
            self.created_files.append(current_file_path)
            shutil.copyfile(self.object_location, current_file_path)


            tree = ET.parse(current_file_path)
            root = tree.getroot()

            for geom in root.findall(".//geom"):
                geom.set("friction", f"{self.friction[i]} 0.0 0.0")
                geom.set("density", str(self.density[i]))
                
            tree.write(current_file_path)

        print("Running Density Optimization...")
        solve_and_update(self.created_files, self.center_of_mass)
        print("Density Optimization Finished.")
        
        return self.created_files


    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.created_files:
            os.remove(file)
