import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import reduce
from typing import Optional, List, Tuple

import numpy as np
from scipy.special import expit

from . import log, colvars
from .utils.io import makedirs
from .utils.trajs import load_traj_for_regex

_log = log.getLogger(__name__)


@dataclass
class IterationRunner(object):
    iteration: int
    swarm_size: int
    exploration_type: str
    cvs: List[colvars.CV]
    seconds_to_sleep: Optional[int] = 3
    query: Optional[str] = "protein"  # "not (resname =~ 'POP') and not water"

    def run(self) -> None:
        self.submit_jobs()
        self.wait_for_completion()
        self.postprocess()

    def submit_jobs(self) -> subprocess.Popen:
        if self.simulations_finished():
            return None
        args = [
            "sbatch",
            "--array=0-{}".format(self.swarm_size - 1),
            "../submit_walkers.sh"
        ]
        return subprocess.Popen(['/bin/bash', '-c', " ".join(args)])

    def wait_for_completion(self) -> bool:
        _log.info("Waiting for completion")
        while not self.simulations_finished():
            time.sleep(self.seconds_to_sleep)
        return True

    def postprocess(self) -> None:
        """Create input for next iteration"""
        _log.info("Postprocessing")
        center, distance_to_center = self.compute_center_distances()
        self.generate_replicas(distance_to_center)

    def simulations_finished(self) -> bool:
        files_missing = False
        for i in range(self.swarm_size):
            if not os.path.exists("s{}.done".format(i)):
                files_missing = True
                break
        return not files_missing

    def compute_center_distances(self) -> Tuple[np.array, np.array]:
        evals = self._load_evals()
        all_evs = reduce(lambda e1, e2: np.append(e1, e2, axis=0), evals)
        print(all_evs.shape, evals[0].shape)
        center = all_evs.mean(axis=0)
        distance_to_center = np.empty((self.swarm_size,))
        for idx, ev in enumerate(evals):
            endpoint = ev[-1]
            distance_to_center[idx] = np.linalg.norm(center - endpoint)
        return center, distance_to_center

    def generate_replicas(self, distance_to_center: np.array) -> None:
        next_iter_dir = "../{}/".format(self.iteration + 1)
        makedirs(next_iter_dir, overwrite=True, backup=True)
        n_replicas = self._compute_number_of_replicas(distance_to_center)
        # Iterate through in descending order
        counter = 0
        for idx, nreps in enumerate(n_replicas):
            infile = "s{}.gro".format(idx)
            _log.info("Trajectory %s will seed %s new replicas", idx, nreps)
            for n in range(nreps):
                shutil.copy(infile, "{}in-{}.gro".format(next_iter_dir, counter))
                counter += 1

        if counter != self.swarm_size:
            _log.error(
                "Reweighting of replicas tried to copy wrong number of replicas (counter=%s). Double check your code.\ndistances: %s\nnreplicas:%s",
                counter, distance_to_center, n_replicas)
            raise OverflowError()

    def _compute_number_of_replicas(self, distance_to_center: np.array) -> np.array:
        mean_dist = distance_to_center.mean()

        def to_weight(d):
            if "single" in self.exploration_type:
                return np.exp(-(d / mean_dist) ** 2)
            elif "multi" in self.exploration_type:
                return expit((d / mean_dist) ** 2)
            else:
                raise ValueError("{} is not a valid exploration type".format(self.exploration_type))

        weights = np.array([to_weight(d) for d in distance_to_center])
        n_replicas = np.zeros(self.swarm_size, dtype=int)
        replica = self.swarm_size
        res = []
        while replica > 0:
            replica = int(replica)
            weights[0:replica] /= weights[0:replica].sum()
            for weight_order, idx in enumerate(reversed(np.argsort(weights))):
                if weight_order >= replica:
                    break
                fractional_replicas = replica * weights[idx]
                # for rounding,
                # see https://stackoverflow.com/questions/28617841/rounding-to-nearest-int-with-numpy-rint-not-consistent-for-5
                n_replicas[idx] += int(np.floor(fractional_replicas) + 0.5)
                if n_replicas.sum() >= self.swarm_size:
                    n_replicas[idx] -= n_replicas.sum() - self.swarm_size
            replica = self.swarm_size - n_replicas.sum()
            res.append(replica)

        return n_replicas

    def _load_evals(self) -> List[np.array]:
        """

        :return: The CV values for every walker trajectory
        """
        evals = []
        for i in range(self.swarm_size):
            t = load_traj_for_regex("./",
                                    "s{}.xtc".format(i),
                                    "s{}.gro".format(i),
                                    stride=1,
                                    query=self.query,
                                    print_files=False)
            ev = colvars.eval_cvs(cvs=self.cvs, traj=t)
            evals.append(ev)
        return evals
