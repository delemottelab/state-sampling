import os
import re
import shutil
import time

from .. import log

_log = log.getLogger("utils-io")


def sorted_alphanumeric(l):
    """
    From https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python/
    Sorts the given iterable in the way that is expected.
    Required arguments:
    l -- The iterable to be sorted.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def read_file(path):
    with open(path, "r") as myfile:
        data = myfile.read()
    return data


def get_backup_name(path):
    return path.strip("/") + "#" + time.strftime('%Y%m%d-%H%M')


def backup_path(oldpath):
    """Backups/Moves the path to a backup name"""
    if os.path.exists(oldpath):
        newpath = get_backup_name(oldpath)
        if os.path.exists(newpath):
            backup_path(newpath)
        shutil.move(oldpath, newpath)


def makedirs(path, overwrite=False, backup=True):
    if os.path.exists(path):
        if overwrite:
            if backup:
                backup_path(path)
            else:
                shutil.rmtree(path)  # beware, removes all the subdirectories!
            os.makedirs(path)
    else:
        os.makedirs(path)


def make_parentdirs(filepath):
    try:
        os.makedirs(os.path.dirname(filepath))
    except Exception as ex:
        pass  # ok, exists already


def copy_and_inject(infile, outfile, params, marker="$%s", start_index=0):
    text = read_file(infile)
    with open(outfile, "w") as out:
        out.write(inject(text, params, marker=marker, start_index=start_index))


def inject(text, params, marker="$%s", start_index=0):
    for i in range(start_index, len(params) + start_index):
        text = text.replace(marker % i, str(params[i - start_index]))
    return text


def escape_to_filename(filename):
    return re.sub(r'[\W]', '', filename)
