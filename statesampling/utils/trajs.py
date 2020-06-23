import glob

import mdtraj as md
import numpy as np

from .io import sorted_alphanumeric
from .. import log

_log = log.getLogger("utils-trajs")


def fix_pbc(traj, check_if_necessary=True, atom_q="name CA and (resSeq 131 or resSeq 268)", epsilon=1e-3):
    """
    Make a test and see if the distances between the atoms is affected by the periodic boundary conditions.
    If so, center the trajectory so that the protein is not broken
    """
    if not check_if_necessary:
        return center_protein(traj)
    atoms = traj.top.select(atom_q)
    d_pbc = md.compute_distances(
        traj,
        [atoms],
        periodic=True)
    d_nopbc = md.compute_distances(
        traj,
        [atoms],
        periodic=False)
    diff = np.absolute(d_pbc - d_nopbc)
    diff = diff[diff > epsilon]
    if len(diff) > 0:
        # We need to align the molecule
        _log.debug("Aligning protein molecule to fix PBCs")
        return center_protein(traj)
    else:
        return traj


def create_bonds(topology):
    if next(topology.bonds, None) is None:
        _log.debug("No bonds in topology, creating standard bonds for alignment")
        topology.create_standard_bonds()


def center_protein(traj, inplace=True):
    """Center protein so that the molecule is not broken by periodic boundary conditions"""
    create_bonds(traj.topology)
    return traj.image_molecules(inplace=inplace, make_whole=True)


def align_frames(traj, query="protein and name CA"):
    atoms = traj.top.select(query)
    return traj.superpose(traj, frame=0, atom_indices=atoms, ref_atom_indices=atoms, parallel=True)


def load_traj_for_regex(directory,
                        traj_filename,
                        top_filename,
                        stride=1,
                        query=None,
                        center_and_align=True,
                        sort_function=sorted_alphanumeric,
                        print_files=False):
    toptraj = md.load(glob.glob(directory + top_filename)[0])
    if query is not None:
        atom_indices = toptraj.top.select(query)
    else:
        atom_indices = None
    if traj_filename is None:
        return toptraj if atom_indices is None else toptraj.atom_slice(atom_indices)
    file_list = sort_function(glob.glob(directory + traj_filename))
    _log.debug("Loading %s files from directory %s", len(file_list), directory)
    if print_files:
        _log.debug("Trajectories included:\n%s", "\n".join([t for t in file_list]))
    traj = md.load(
        file_list,
        top=toptraj.top,
        atom_indices=atom_indices,
        stride=stride)
    if center_and_align:
        traj = fix_pbc(traj)
        traj = align_frames(traj)
    return traj
