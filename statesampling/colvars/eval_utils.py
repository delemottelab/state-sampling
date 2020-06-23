import numpy as np


def eval_cvs(cvs, traj, rescale=False):
    res = np.empty((len(traj), len(cvs)))
    for i, cv in enumerate(cvs):
        res[:, i] = np.squeeze(cv.eval(traj))
    return rescale_evals(res, cvs) if rescale else res


def rescale_evals(cvs, evals):
    if len(evals.shape) == 1:
        return cvs[0].rescale(evals)
    res = np.empty(evals.shape)
    for i, cv in enumerate(cvs):
        res[:, i] = cv.rescale(evals[:, i])
    return res


def scale_evals(cvs, evals):
    """The opposite of rescale_evals"""
    if len(evals.shape) == 1:
        return cvs[0].scale(evals)
    res = np.empty(evals.shape)
    for i, cv in enumerate(cvs):
        res[:, i] = cv.scale(evals[:, i])
    return res


def normalize_cvs(cvs, simulations=None, trajs=None):
    if simulations is not None and trajs is None:
        trajs = [s.traj for s in simulations]
    if trajs is not None:
        for cv in cvs:
            cv.normalize(trajs)
    return cvs


def rescale_points(cvs, points):
    if len(points.shape) == 1:
        return np.array([cv.rescale(p) for cv, p in zip(cvs, points)])
    else:
        res = np.empty(points.shape)
        for i, point in enumerate(points):
            for j, cv in enumerate(cvs):
                res[i, j] = cv.rescale(point[j])
        return res


def scale_points(cvs, points):
    """THe opposite of rescale_points"""
    if len(points.shape) == 1:
        return np.array([cv.scale(p) for cv, p in zip(cvs, points)])
    else:
        res = np.empty(points.shape)
        for i, point in enumerate(points):
            for j, cv in enumerate(cvs):
                res[i, j] = cv.scale(point[j])
        return res
