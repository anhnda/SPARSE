import params


def genDataByPref(pref=""):
    r"""
    The wrapper for generating k-fold data
    Args:
        pref: either "" for TWOSIDES, "c" for CADDDI, and "j" for JADERDDI

    """
    params.PRINT_DB = False
    if pref == "" or pref.lower() == "t":
        params.D_PREF = ""
        __genTWOSIDES()
    elif pref.lower() == "c":
        __genCAD()
    elif pref.lower() == "j":
        __genJADER()
    else:
        print("Error: Unknown data with prefix: ", pref)
        exit(-1)


def __genTWOSIDES():
    from dataFactory.genData.genTWOSIDES import run
    # Generate both full high quality TWOSIDES and K-Folds TWOSIDES
    params.HIGH_TWOSIDES = True
    run()
    params.HIGH_TWOSIDES = False
    run()


def __genJADER():
    from dataFactory.genData.genJADER import run
    run()


def __genCAD():
    from dataFactory.genData.genCAD import run
    run()


