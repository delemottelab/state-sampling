import json
import os
from typing import Dict, Any, List

import numpy as np

from .. import log
import statesampling.utils as utils
from .cvs import CV, InverseContactCv, ContactCv

_log = log.getLogger("colvars-io")


def load_cvs_definition(filepath: str) -> Dict[str, Any]:
    """
    :param filepath:
    :return json objects
    """
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def load_cvs(filepath: str) -> np.array:
    """
    Loads a file and returns CV objects
    :param filepath:
    :return:
    """
    defs = load_cvs_definition(filepath)
    return create_cvs(defs)


def create_cvs(cvs_definition: Dict[str, Any]) -> np.array:
    """
    Create cv objects from definitions of standard types such as float, strings and ints defined as json objects
    :param cvs_definition: JSONs as loaded by function load_cvs_definition
    :return: an numpy array of cvs.py
    """
    cvs = []
    for i, cv_def in enumerate(cvs_definition["cvs"]):
        clazz = cv_def["@class"]
        if clazz == InverseContactCv.__name__:
            cv = _parse_InverseContactCv(cv_def)
        elif clazz == ContactCv.__name__:
            cv = _parse_ContactCv(cv_def)
        else:
            _log.warn("Class %s cannot be parsed right now (index %s, json %s). Please implement.", clazz, i, cv_def)
            continue
        cv.name = cv_def.get("name", cv.id)
        cv.importance = cv_def.get("importance", None)
        cv.normalize(scale=cv_def.get("scale", 1.), offset=cv_def.get("offset", 0.))
        cvs.append(cv)

    return np.array(cvs)


def save_cvs_definitions(filepath: str, cvs_definition: Dict[str, Any]) -> None:
    """
    :param filepath:
    :param cvs_definition - JSONs as loaded by function load_cvs_definition
    :return:
    """
    dir = os.path.dirname(filepath)
    utils.io.makedirs(dir, overwrite=False, backup=True)
    with open(filepath, 'w') as outfile:
        outfile.write(cvs_definition)


def save_cvs(filepath: str, cvs: List[CV]) -> None:
    """
    :param filepath:
    :param cvs
    :return:
    """
    cvs_definition = create_cvs_definitions(cvs)
    save_cvs_definitions(filepath, cvs_definition)


def create_cvs_definitions(cvs: List[CV]) -> Dict[str, Any]:
    """
    Create JSONs(dicts) from CVs
    :param cvs
    :return: an numpy array of py
    """
    cvs_def = []
    for cv in cvs:
        clazz = cv.__class__
        cv_def = {
            "@class": cv.__class__.__name__,
            "id": cv.id,
            "name": cv.name,
            "scale": float(cv.norm_scale),
            "offset": float(cv.norm_offset),
            "importance": cv.importance if hasattr(cv, "importance") else None

        }
        if clazz == ContactCv or clazz == InverseContactCv:  # InveseContact and Contact use same serializer
            _serialize_ContactCv(cv, cv_def)
        else:
            _log.warn("Class %s cannot be parsed right now (cv). Please implement.", clazz, cv)
            continue
        cvs_def.append(cv_def)
    cvs_def = {"cvs": cvs_def}
    return json.dumps(cvs_def, ensure_ascii=False, indent=2)


def _serialize_ContactCv(cv, cv_def):
    cv_def.update({
        "res1": cv.res1,
        "res2": cv.res2,
        "scheme": cv.scheme,
        "periodic": cv.periodic
    })


def _parse_InverseContactCv(cv_def):
    cv = InverseContactCv(ID=cv_def["id"], res1=cv_def["res1"], res2=cv_def["res2"],
                          scheme=cv_def.get("scheme", "closest-heavy"),
                          periodic=cv_def.get("periodic", True))
    return cv


def _parse_ContactCv(cv_def):
    cv = ContactCv(ID=cv_def["id"], res1=cv_def["res1"], res2=cv_def["res2"],
                   scheme=cv_def.get("scheme", "closest-heavy"),
                   periodic=cv_def.get("periodic", True))
    return cv
