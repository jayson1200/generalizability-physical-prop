import os
import xml.etree.ElementTree as ET


def fix_density(path):
    tree = ET.parse(path)
    root = tree.getroot()
    for elem in root.iter():
        if "density" in elem.attrib:
            elem.attrib["density"] = "1000"

    base_path = os.path.dirname(path)
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    density_path = os.path.join(base_path, f"{name}_density{ext}")
    tree.write(density_path)
    return density_path
