from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-colorblind")

from .. import log

_log = log.getLogger("utils-viz")


def show_convergence(center_points: List, outfile: Optional[str] = None) -> None:
    npoints = len(center_points)
    convergence = np.empty((npoints - 1))
    previous_point = None
    xticks = []
    for idx, cp in enumerate(center_points):
        if previous_point is not None:
            convergence[idx - 1] = np.linalg.norm(previous_point - cp)
            xticks.append("$|\\bar{c}_{%s}-\\bar{c}_{%s}|$" % (idx, idx - 1))
        else:
            xticks.append("$|\\bar{c}_{%s}-\\bar{c}_{in}|$" % (idx))
        previous_point = cp
    xvals = np.linspace(1, npoints - 1, npoints - 1)
    plt.plot(xvals, convergence)
    plt.xlabel("Iteration")
    plt.xticks(xvals, xticks)
    plt.ylabel("Distance between iteration centers")
    plt.tight_layout(pad=0.3)
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
