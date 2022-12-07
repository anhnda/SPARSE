import params


def loadDrugBankX():
    fin = open(params.NEW_DRUGBANK_X)
    d = {}
    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip().lower()
        parts = line.split("||")
        hardName = parts[0]
        synLine = parts[-2]
        saltLine = parts[-1]
        if len(synLine) > 1:
            syns = synLine.split(",")
        else:
            syns = []
        if len(saltLine) > 1:
            salts = saltLine.split(",")
        else:
            salts = []
        d[hardName] = hardName
        for ll in [syns, salts]:
            for l in ll:
                d[l] = hardName
    return d


def filterDrugName():
    from utils import utils
    fin = open("%s2" % params.SELECTED_DRUGS_INPUT)
    fout = open(params.SELECTED_DRUGS_FILTERED, "w")
    d = loadDrugBankX()
    drugINs = fin.readlines()
    for drug in drugINs:
        dmap = utils.get_dict(d, drug.lower().strip(), -1)
        if dmap != -1:
            fout.write("%s\n" % dmap)
    fout.close()


if __name__ == "__main__":
    filterDrugName()
