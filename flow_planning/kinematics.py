"""Franka FK via pytorch_kinematics, from the same URDF the sim loads.
Validated to match newton's eval_fk to floating-point precision."""

import re
from pathlib import Path

import torch
from torch import Tensor

EE_FRAME = "fr3_hand_tcp"


def build_franka_chain(device: str):
    """Differentiable franka kinematic chain from the sim's URDF. Returns
    (chain, njoints). The fingers' visual geometry is stripped (the hand mesh
    covers the gripper); collision STLs serve as the link meshes for the SDF."""
    import newton.utils
    import pytorch_kinematics as pk
    from pytorch_kinematics.urdf_parser_py.xml_reflection import core as xml_core

    # silence stderr spam for vendor URDF extensions the parser ignores anyway
    xml_core.on_error = lambda msg: None  # ty: ignore[invalid-assignment]

    asset = newton.utils.download_asset("franka_emika_panda")
    urdf = (asset / "urdf/fr3_franka_hand.urdf").read_text()
    for link in ("fr3_leftfinger", "fr3_rightfinger"):
        m = re.search(rf'(<link name="{link}">)(.*?)(</link>)', urdf, re.S)
        assert m is not None, f"link {link} not found in URDF"
        body = re.sub(r"<visual[^>]*>.*?</visual>", "", m.group(2), flags=re.S)
        urdf = urdf[: m.start()] + m.group(1) + body + m.group(3) + urdf[m.end() :]
    urdf = urdf.replace("package://franka_emika_panda/", "")
    # RobotSDF reads link "visual" geometry; point it at the collision STLs
    urdf = urdf.replace("/visual/", "/collision/").replace(".dae", ".stl")
    chain = pk.build_chain_from_urdf(urdf.encode())
    chain = chain.to(dtype=torch.float32, device=device)
    return chain, len(chain.get_joint_parameter_names()), str(asset) + "/"


def ee_positions(chain, q: Tensor) -> Tensor:
    """q: (B, n_arm) arm joint angles -> (B, 3) EE position in the base frame."""
    njoints = len(chain.get_joint_parameter_names())
    pad = q.new_zeros(q.shape[0], njoints - q.shape[1])
    fk = chain.forward_kinematics(torch.cat([q, pad], dim=1))
    return fk[EE_FRAME].get_matrix()[:, :3, 3]


def build_piper_chain(device: str):
    import pytorch_kinematics as pk

    urdf = (Path(__file__).parent / "assets/piper.urdf").read_bytes()
    chain = pk.SerialChain(pk.build_chain_from_urdf(urdf), "link6")
    return chain.to(dtype=torch.float32, device=device)
