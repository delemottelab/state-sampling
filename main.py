import argparse
import os

from statesampling import log
from statesampling.colvars import eval_cvs
from statesampling.colvars.io import load_cvs
from statesampling.iteration_runner import IterationRunner
from statesampling.utils.io import makedirs
from statesampling.utils.trajs import load_traj_for_regex
from statesampling.utils.visualization import show_convergence

_log = log.getLogger("main")


def start(args):
    _log.info("Using args %s", args)
    start_mode = args.start_mode
    iteration = args.iteration
    cvs = load_cvs(args.cvs)
    cwd = os.getcwd() + "/" + args.working_dir + "/"
    starting_structure = load_traj_for_regex(directory=cwd,
                                             traj_filename=None,
                                             top_filename=args.starting_structure)
    center_points = [eval_cvs(cvs, starting_structure).squeeze()]
    while iteration <= args.max_iteration:
        wd = cwd + str(iteration)
        runner = IterationRunner(iteration=iteration,
                                 swarm_size=args.swarm_size,
                                 cvs=cvs,
                                 exploration_type=args.exploration_type)
        if start_mode == "server":
            if runner.simulations_finished():
                _log.info("Simulation already finished from before.")
                #iteration += 1
                #continue
            makedirs(wd, overwrite=False)
            _log.info("Changing directory to %s", wd)
            os.chdir(wd)
            runner.run()

        elif start_mode == "convergence":
            if not os.path.exists(wd):
                _log.info("No simulation files for iteration {}. Breaking".format(iteration))
                break
            os.chdir(wd)
            if not runner.simulations_finished():
                _log.info("Simulation not finished for iteration {}. Breaking".format(iteration))
                break
        else:
            raise NotImplementedError("Start mode {} nor supported".format(start_mode))
        _log.info("Computing convergence")
        center, _ = runner.compute_center_distances()
        center_points.append(center)
        _log.info("Finished with iteration %s.", iteration)
        iteration += 1

    show_convergence(center_points, outfile="{}/convergence_{}.png".format(cwd, args.simu_id))
    _log.info("Max iteration reached. Finished")


def create_argparser():
    p = argparse.ArgumentParser(
        epilog='State sampling code. Intended to dispatch bash jobs and analyze the resulting trajectories iteratively.\nBy Oliver Fleetwood 2019.')
    p.add_argument('--iteration', type=int, help='String Iteration', required=True)
    p.add_argument('--simu_id', type=str, help='ID to identify this simu', required=False, default="ss")
    p.add_argument('--starting_structure', type=str, required=False, default="equilibrated.gro")
    p.add_argument('--cvs', type=str, help='Path to CVs file', required=False, default="cvs.json")
    p.add_argument('--start_mode', type=str, help="Start mode ('server' or 'convergence')", default="server")
    p.add_argument('--exploration_type', type=str, help="Type of exploration ('single_state' or 'multi_state')",
                   default="single_state")
    p.add_argument('--working_dir', type=str, help='working directory', required=True)
    p.add_argument('--max_iteration', type=int, help='Maximum iteration number or the job will finish',
                   required=False,
                   default=15)  # Fairly low so that you make sure to check that everything is working as expected
    p.add_argument('--swarm_size', type=int,
                   help='Number of trajectories every iteration',
                   default=24)
    return p


if __name__ == '__main__':
    _log.info("----------------Starting state sampling simulation. Code by Oliver Fleetwood 2020------------")
    p = create_argparser()
    start(p.parse_args())
