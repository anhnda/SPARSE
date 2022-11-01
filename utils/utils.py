from io import open

import numpy as np
import joblib
import os
import time


def getCurrentTimeString():
    r"""

    Returns:
        Current timestamp as a string of Year-Month-Day Hour:Minutes:Second
    """
    t = time.localtime()
    currentTime = time.strftime("%Y-%m-%d %H:%M:%S", t)
    return currentTime


def ensure_dir(directory):
    r"""
    Create directory if not exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def convertHexToBinString888(hexString):
    r"""
    Convert Input Hex string to a Binary array
    Args:
        hexString: input hex string
    Returns:
        a binary array with size 888

    """
    # scale = 16  ## equals to hexadecimal
    # num_of_bits = 888
    return bin(int(hexString, 16))[2:].zfill(888)


def convertBinString888ToArray(binString888):
    r"""
    Convert a binary string of size 888 to an numpy array
    Args:
        binString888: binary string

    Returns:
        np array of size 888

    """
    ar = np.ndarray(888, dtype=float)
    ar.fill(0)
    for i in range(887, -1, -1):
        if binString888[i] == "1":
            ar[i] = 1
    return ar


def convertHex888ToArray(hex888):
    r"""
    Convert an input hex string to numpy array of size 888
    Args:
        hex888: input hex string

    Returns:
        numpy array of size 888
    """
    return convertBinString888ToArray(convertHexToBinString888(hex888))


def get_dict(d, k, v=-1):
    r"""
    Get the corresponding value of key k in dictionary d
    If k is not in d then return the default value v
    Args:
        d: dictionary
        k: key
        v: default return value for non-exist key k in d

    Returns:
        d[k] if k in d or v for otherwise

    """
    try:
        v = d[k]
    except:
        pass
    return v


def get_insert_key_dict(d, k, v=0):
    r"""
    Get corresponding value for key k in dictionary k

    Args:
        d: dictionary
        k: key
        v: default value

    Returns:
        d[k] for k in d
        otherwise:
        d[k] = v, return v

    """
    try:
        v = d[k]
    except:
        d[k] = v
    return v


def add_dict_counter(d, k, v=1):
    r"""
    Add value v to the corresponding counter of key k in d
    Args:
        d: dictionary
        k: key
        v: value



    """
    try:
        v0 = d[k]
    except:
        v0 = 0
    d[k] = v0 + v


def sort_dict(dd):
    r"""
    Sort the counter dictionary {key: int_counter_values} by the counter values
    in a descending order
    Args:
        dd: dictionary {key:int_counter_value}

    Returns:
        [list_of_(key,counter) for counter is in a descending order]

    """
    kvs = []
    for key, value in sorted(dd.items(), key=lambda p: (p[1], p[0])):
        kvs.append([key, value])
    return kvs[::-1]


def sum_sort_dict_counter(dd):
    r"""
    Return sum of counter in a counter dict
    Args:
        dd: [list_of_(key,counter) for counter is in a descending order]
    Returns:
        sum of key_counter

    """
    cc = 0
    for p in dd:
        cc += p[1]
    return cc


def get_update_dict_index(d, k):
    r"""
    Get the corresponding index of k in d
        if k is not in d then assign
        the index of k to the length of d
        ortherwise return the corresponding value of k in k

    Args:
        d: dictionary
        k: k

    Returns:
        corresponding index of k

    """
    try:
        current_index = d[k]
    except:
        current_index = len(d)
        d[k] = current_index
    return current_index


def get_dict_index_only(d, k):
    r"""
    Get the corresponding index of k in d
    Do not insert k to d if not exist (return -1)
    Args:
        d: dictionary
        k: key

    Returns:
        corresponding index of k in d

    """
    try:
        current_index = d[k]
    except:
        current_index = -1

    return current_index


def load_list_from_file(path):
    r"""
    Return a list line by line from an input file
    Args:
        path: input file

    Returns: list of line

    """
    list = []
    fin = open(path)
    while True:
        line = fin.readline()
        if line == "":
            break
        list.append(line.strip())
    fin.close()
    return list


def reverse_dict(d):
    r"""
    Swap key-value to value-key in dictionary d
    Args:
        d: dictionary {key:value}

    Returns:
        dictionary {value:key}

    """
    d2 = dict()
    for k, v in d.items():
        d2[v] = k
    return d2


def save_obj(obj, path):
    r"""
    Save object
    Args:
        obj: object to save
        path: path to save

    """
    joblib.dump(obj, path)


def load_obj(path):
    r"""
    Load an object from path
    Args:
        path: path of the saved object

    Returns:

    """
    return joblib.load(path)


def loadMapFromFile(path, sep="\t", keyPos=0, valuePos=1):
    r"""
    Load map from a file with given seperator, positions of key and value
    Args:
        path: path to file
        sep: separator
        keyPos: position of key
        valuePos: position of value

    Returns:
        dictionary {key:value}
    """
    fin = open(path)
    d = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split(sep)
        d[parts[keyPos]] = parts[valuePos]
    fin.close()
    return d


def loadMapSetFromFile(path, sep="\t", keyPos=0, valuePos=1, sepValue="", isStop=""):
    r"""
    Load map from a file with given seperator, key pos, value pos, separator for value
    Args:
        path: path to the file
        sep: separator for key-value
        keyPos: position of key
        valuePos: position of value
        sepValue: seperators of value
        isStop: Indicator to break to two parts: train-test

    Returns:
        two dictionaries dTrain, dTest in the format:
         {key:set_of_corresponding_value}

    """
    fin = open(path)
    dTrain = dict()

    if isStop != "":
        dTest = dict()

    d = dTrain

    while True:
        line = fin.readline()
        if line == "":
            break
        if isStop != "":
            if line.startswith(isStop):
                d = dTest
                continue
        parts = line.strip().split(sep)
        v = get_insert_key_dict(d, parts[keyPos], set())
        if sepValue == "":
            v.add(parts[valuePos])
        else:
            values = parts[valuePos]
            values = values.split(sepValue)
            for value in values:
                v.add(value)
    fin.close()
    if isStop != "":
        return dTrain, dTest
    return dTrain

