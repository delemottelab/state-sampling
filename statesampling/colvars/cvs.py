from typing import Optional, Callable, Any, List, Tuple

import mdtraj as md
import numpy as np
from dataclasses import dataclass

from .. import log

_log = log.getLogger(__name__)


@dataclass
class CV(object):
    """
    Collective variable class with an id and a generator to convert every frame in a trajectory to a numerical value
    """
    ID: str
    name: Optional[str] = None
    generator: Optional[Callable[[Any], np.array]] = None
    norm_offset: Optional[float] = 0
    norm_scale: Optional[float] = 1
    importance: Optional[float] = 1

    def __post_init__(self):
        if self.name is None:
            self.name = self.ID

    @property
    def id(self) -> str:
        return self.ID

    def __str__(self):
        return self.ID if self.name is None else self.name

    def normalize(self, trajs: Optional[List] = None, scale=1, offset=0) -> Tuple[float, float]:
        """
        Scales the CV according to the following value (eval(traj)-offset)/scale

        Your CVs should be normalized to values between 0 and 1,
        meaning that 'scale' is the difference between the max and min value along the traj
        and 'offset' is the minimum value
        """
        if trajs is not None and len(trajs) > 0:
            max_val, min_val = None, None
            for t in trajs:
                if self.generator is None:
                    raise Exception("Field 'generator' has not been set")
                evals = self.generator(t)
                tmax = evals.max()
                tmin = evals.min()
                if max_val is None or max_val < tmax:
                    max_val = tmax
                if min_val is None or min_val > tmin:
                    min_val = tmin
            scale = max_val - min_val if max_val > min_val + 1e-4 else max_val
            offset = min_val
        self.norm_offset = offset
        self.norm_scale = scale
        return scale, offset

    def rescale(self, point):
        """Scales back to the original physical values"""
        return self.norm_scale * point + self.norm_offset

    def scale(self, point):
        """Scales to the normalized values"""
        return (point - self.norm_offset) / self.norm_scale

    def eval(self, traj) -> np.array:
        if self.generator is None:
            raise Exception("Field 'generator' has not been set")
        return (self.generator(traj) - self.norm_offset) / self.norm_scale


@dataclass
class ContactCv(CV):
    res1: Optional[int] = -1
    res2: Optional[int] = -1
    scheme: Optional[str] = "closest-heavy"
    periodic: Optional[bool] = True

    def __post_init__(self):
        self.generator = self.compute_contact

    def compute_contact(self, traj):
        res1_idx, res2_idx = None, None
        for residue in traj.topology.residues:
            if residue.is_protein:
                if residue.resSeq == self.res1:
                    res1_idx = residue.index
                    if res2_idx is not None and res2_idx > -1:
                        break
                elif residue.resSeq == self.res2:
                    res2_idx = residue.index
                    if res1_idx is not None and res1_idx > -1:
                        break
        if res1_idx is None:
            raise ValueError("No residue with id {}".format(self.res1))
        if res2_idx is None:
            raise ValueError("No residue with id {}".format(self.res2))
        dists, atoms = md.compute_contacts(traj, contacts=[[res1_idx, res2_idx]], scheme=self.scheme,
                                           periodic=self.periodic)
        return dists


@dataclass
class InverseContactCv(ContactCv):
    """
    Same as ContactCv but with the inverse of the distance
    """

    def compute_contact(self, traj):
        return 1 / ContactCv.compute_contact(self, traj)


@dataclass()
class RmsdCv(CV):
    """
    Class for RMSD based CVs to a reference structure
    """
    query: Optional[str] = "protein and element != 'H'"
    reference_structure: Optional = None
    warn_missing_atoms: Optional[bool] = True

    def __post_init__(self):
        self.generator = self.compute_rmsd

    def compute_rmsd(self, traj):
        simu_atoms, ref_atoms = self.select_atoms_incommon(traj.topology)
        rmsds = md.rmsd(traj.atom_slice(simu_atoms), self.reference_structure.atom_slice(ref_atoms))
        return rmsds

    def select_atoms_incommon(self, top):
        """
        Matches atoms returned by the query for both topologies by name and returns the atom indices for the respective topology
        """
        atoms = [top.atom(idx) for idx in top.select(self.query)]
        ref_atoms = [self.reference_structure.top.atom(idx) for idx in self.reference_structure.top.select(self.query)]
        ref_atoms, missing_atoms = _filter_atoms(ref_atoms, atoms)
        if self.warn_missing_atoms and len(missing_atoms) > 0:
            _log.warn("%s atoms in reference not found topology. They will be ignored. %s", len(
                missing_atoms), missing_atoms)
        atoms, missing_atoms = _filter_atoms(atoms, ref_atoms)
        if self.warn_missing_atoms and len(missing_atoms) > 0:
            _log.warn("%s atoms in topology not found reference. They will be ignored. %s", len(
                missing_atoms), missing_atoms)
        duplicate_atoms = _find_duplicates(atoms)
        if self.warn_missing_atoms and len(duplicate_atoms) > 0:
            _log.warn("%s duplicates found in topology %s", len(duplicate_atoms), duplicate_atoms)
        duplicate_atoms = _find_duplicates(ref_atoms)
        if self.warn_missing_atoms and len(duplicate_atoms) > 0:
            _log.warn("%s duplicates found in reference %s", len(duplicate_atoms), duplicate_atoms)
        if self.warn_missing_atoms and len(atoms) != len(ref_atoms):
            _log.warn("number of atoms in result differ: %s vs %s",
                      len(atoms), len(ref_atoms))
        return [a.index for a in atoms], [a.index for a in ref_atoms]


def _find_duplicates(atoms):
    atom_names = [str(a) for a in atoms]
    return [a for a in atoms if atom_names.count(str(a)) > 1]


def _filter_atoms(atoms, ref_atoms):
    """
    Returns atoms which name matched the name i ref_atoms as well as the once which did not match.
    Matching is done on name, i.e. str(atom)
    TODO speed up with a search tree or hashmap
    """
    ref_atom_names = [str(a) for a in ref_atoms]
    missing_atoms = []
    matching_atoms = []
    # Atoms in inactive not in simu
    for atom in atoms:
        if str(atom) not in ref_atom_names:
            # print(atom)
            missing_atoms.append(atom)
        else:
            # print("FOUND IT", atom)
            matching_atoms.append(atom)
    return matching_atoms, missing_atoms
