"""
    This is not library quality code
"""
import shutil

import trimesh
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import minimize
import os

COM = "com"
FRICTION = "friction"
SCALE = "scale"
DENSITY = "density"

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
        name = m.get("name")
        fname = m.get("file")
        if name and fname:
            mesh_assets[name] = fname

    return mesh_assets

def set_mass_distribution(model_xml_location, target_center_of_mass_position : np.ndarray):
    tree = ET.parse(model_xml_location)
    root = tree.getroot()

    mesh_assets = get_mesh_info(model_xml_location)

    geoms = root.findall(".//geom")

    mean_positions, volumes = zip(*[get_info(os.path.join(os.path.dirname(model_xml_location), mesh_assets[geom.get("mesh")])) for geom in geoms])
    densities = [float(geom.get("density")) for geom in geoms]

    volumes_np = np.array(volumes)
    densities_np = np.array(densities)

    mass_np = volumes_np * densities_np
    total_mass = np.sum(mass_np)

    MASS_MAT = (1/total_mass) * np.array(mean_positions).T * mass_np.reshape(1, -1)

    def objective(x):
        r = MASS_MAT.dot(x) - target_center_of_mass_position
        return np.dot(r, r) 
    
    def grad(x):
        return 2 * MASS_MAT.T @ (MASS_MAT @ x  - target_center_of_mass_position)

    # constraints & bounds
    cons = {
        'type': 'eq',
        'fun': lambda x: np.dot(mass_np, x) - total_mass,
        'jac': lambda x: mass_np
    }

    bounds = [(0, None)] * len(volumes)

    x0 = np.random.rand(len(mass_np))
    x0 *= total_mass / (mass_np @ x0)


    res = minimize(objective, x0, jac=grad,
                   bounds=bounds,
                   constraints=cons,
                   method='SLSQP',
                   options={'ftol': 1e-6})

    if not res.success:
        print("Failed to find optimal mass distribution. Weakening constraints and trying again")

        res = minimize(objective, x0, jac=grad,
               bounds=bounds,
               constraints=cons,
               method='SLSQP',
               options={'ftol': 5e-5})

        if not res.success:
            raise Exception("Failed to find optimal mass distribution")

    optimal_mass_scales = res.x
    
    final_error = MASS_MAT.dot(optimal_mass_scales) - target_center_of_mass_position
    final_error = np.linalg.norm(final_error)
    
    print(f"Distance from solution: {final_error}")

    optimal_mass = optimal_mass_scales * mass_np
    optimal_densities = optimal_mass / volumes_np

    for geom, density in zip(geoms, optimal_densities):
        geom.set("density", str(max(density, 0.00005)))

    tree.write(model_xml_location)


class MJCFHandler:
    def __init__(self, command_args):
        self.tmp_location = None
        self.object_location = None
        self.model_path = command_args.obj_path
        self.command_args = command_args

        self.target = None

        if command_args.friction:
            self.target = FRICTION
        elif command_args.density:
            self.target = DENSITY
        elif command_args.center_of_mass:
            self.target = COM
        elif command_args.scale:
            self.target = SCALE


    def __enter__(self):
        self.object_location = self.model_path
        self.tmp_location =  os.path.dirname(self.model_path) + "/model_backup.xml"

        shutil.copyfile(self.object_location, self.tmp_location)

        tree = ET.parse(self.object_location)
        root = tree.getroot()
        ret = {}

        for geom in root.findall(".//geom"):
            if self.command_args.friction:
                geom.set("friction", self.command_args.friction)
                ret["friction"] = self.command_args.friction
            elif self.command_args.density:
                ret["density"] = self.command_args.density
                geom.set("density", self.command_args.density)
            elif self.command_args.scale:
                ret["scale"] = self.command_args.scale      
        
        tree.write(self.object_location)

        if self.command_args.center_of_mass:
            set_mass_distribution(self.object_location,
                                  np.array(self.command_args.center_of_mass.split(), dtype=float))
            ret["com"] = self.command_args.center_of_mass

        viz_name = ''.join(f"{k}{v}" for k, v in ret.items())
        
        return "no-peturb" if viz_name == '' else viz_name 

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.copyfile(self.tmp_location, self.object_location)
        os.remove(self.tmp_location)
