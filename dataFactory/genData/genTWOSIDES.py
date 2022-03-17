import dataFactory.dataLoader
import params
from utils import utils
import random
import copy
from dataFactory.moleculeFactory import MoleculeFactory

from multiprocessing import Process, Value, Queue

from dataFactory.dataLoader import RealData, RealFoldData
import itertools
import time
import numpy as np
import torch

DATASET_DIR = params.DDI_DIR
params.SAMPLE_NEG = 1300  # Proportion to the number of positive samples for each drug pair of the dataset
DATASET_DIR = "%s/TWOSIDES" % params.TMP_DIR
utils.ensure_dir(DATASET_DIR)
DUMP_FILE = "%s/dump.pkl" % DATASET_DIR


def print_db(*msg):
    if params.PRINT_DB:
        print(*msg)


def resetRandomSeed():
    import random
    random.seed(params.TORCH_SEED)
    torch.manual_seed(params.TORCH_SEED)
    np.random.seed(params.TORCH_SEED)


def loadPubChem():
    return utils.load_obj(params.PUBCHEM_FILE)


def loadMonoADR():
    fin = open(params.MONO_ADR_FILE)
    dDrug2ADRSet = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("|")
        inchi = parts[1]
        adrs = parts[2]
        adrSet = set()
        for adr in adrs:
            adrSet.add(adr)
        dDrug2ADRSet[inchi] = adrSet
    fin.close()
    return dDrug2ADRSet


def loadProtein2Pathway():
    fin = open("%s/KEGG/uniprot_2_pathway.txt" % params.DATA_DIR)
    dPathway2Id = dict()
    dProtein2Pathways = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("|")
        protein = parts[0]
        pathways = parts[1].split(",")
        ar = [utils.get_update_dict_index(dPathway2Id, pathway) for pathway in pathways]
        dProtein2Pathways[protein] = ar
    fin.close()
    dPathway2Name = dict()
    fin = open("%s/KEGG/path:hsa.txt" % params.DATA_DIR)
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        dPathway2Name[parts[0]] = parts[1]
    fin.close()
    return dProtein2Pathways, dPathway2Id, dPathway2Name


def loadDrug2Protein(inchies, pathway=True):
    dInchi2Id = dict()
    for inchi in inchies:
        utils.get_update_dict_index(dInchi2Id, inchi)
    nDrug = len(dInchi2Id)
    drug2ProteinList = dataFactory.dataLoader.loadDrugProteinMap()
    # print(drug2ProteinList['ILVYCEVXHALBSC-OTBYEXOQSA-N'])
    proteinListList = sorted(list(drug2ProteinList.values()))
    protensSets = set()
    protein2Id = dict()
    dProtein2Pathways, dPathway2Id, _ = loadProtein2Pathway()

    for proteins in proteinListList:
        for protein in proteins:
            if protein != "":
                protensSets.add(protein)

    proteinList = list(protensSets)
    proteinList = sorted(proteinList)
    for protein in proteinList:
        utils.get_update_dict_index(protein2Id, protein)

    dDrug2ProteinFeatures = dict()
    nP = len(protein2Id)
    nA = len(dPathway2Id)
    nS = nP
    if pathway:
        nS += nA
    edge_index = []
    cc = 0
    for drugInchi, proteins in drug2ProteinList.items():
        drugId = utils.get_dict(dInchi2Id, drugInchi, -1)
        if drugId == -1:
            cc += 1
            continue
        proteinFeature = np.zeros(nS)
        for p in proteins:
            if p == "":
                continue
            piD0 = protein2Id[p]
            proteinFeature[piD0] = 1
            if pathway:
                pa = utils.get_dict(dProtein2Pathways, p, [])
                for a in pa:
                    proteinFeature[a + nP] = 1

            pId = piD0 + nDrug
            edge_index.append([drugId, pId])
            edge_index.append([pId, drugId])
        dDrug2ProteinFeatures[drugInchi] = proteinFeature
    return edge_index, protein2Id, nDrug, dDrug2ProteinFeatures


def appendProteinProtein(protein2Id, edg_index, nDrug):
    fin = open(params.PPI_FILE)
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        p1 = utils.get_dict(protein2Id, parts[0], -1)
        p2 = utils.get_dict(protein2Id, parts[1], -1)
        if p1 != -1 and p2 != -1:
            edg_index.append([p1 + nDrug, p2 + nDrug])
            edg_index.append([p2 + nDrug, p1 + nDrug])

    fin.close()
    return edg_index


def loadInchi2SMILE():
    f = open(params.DRUGBANK_ATC_INCHI)
    inchi2SMILE = dict()
    while True:
        line = f.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        inchi2SMILE[parts[-1]] = parts[4]
    f.close()
    return inchi2SMILE


def createSubSet(inp_path=params.PATH_TWOSIDES_A):
    def loadInvalidSes():
        return set(line.strip() for line in open("%s/exception_adrs.txt" % params.DDI_DIR).readlines())

    inchi2FingerPrint = loadPubChem()
    inchiKeys = inchi2FingerPrint.keys()

    monoADR = loadMonoADR()
    fin = open(inp_path)
    drugSet = set()
    adrSet = set()
    drugCount = dict()
    adrCount = dict()
    drugPair2ADR = dict()
    inchi2Drug = dict()

    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("|")

        d1, d2, inchi1, inchi2 = parts[0], parts[1], parts[2], parts[3]
        if inchi1 not in inchiKeys or inchi2 not in inchiKeys:
            continue
        adrs = parts[4].split(",")
        inchi2Drug[inchi1] = d1
        inchi2Drug[inchi2] = d2
        drugSet.add(inchi1)
        drugSet.add(inchi2)
        utils.add_dict_counter(drugCount, inchi1)
        utils.add_dict_counter(drugCount, inchi2)
        utils.get_dict(monoADR, inchi1, set())
        utils.get_dict(monoADR, inchi2, set())
        for adr in adrs:
            adrSet.add(adr)
            utils.add_dict_counter(adrCount, adr)
        drugPair2ADR[(inchi1, inchi2)] = adrs

    fin.close()

    adrCountsSorted = utils.sort_dict(adrCount)
    cc = []
    for p in adrCountsSorted:
        _, v = p
        cc.append(v)

    validADRs = set()
    invalidSe = loadInvalidSes()
    endADR = min(len(adrCountsSorted), params.ADR_OFFSET + params.MAX_R_ADR)
    orderedADR = list()
    for i in range(params.ADR_OFFSET, endADR):
        adr, _ = adrCountsSorted[i]
        isValid = True
        for xx in invalidSe:
            if adr.__contains__(xx):
                isValid = False
                break
        if not isValid:
            continue
        validADRs.add(adr)
        orderedADR.append(adr)

    drugCountSorted = utils.sort_dict(drugCount)
    validInchi = set()
    m = min(len(drugCount), params.MAX_R_DRUG)
    MIN_D = 10
    for i in range(m):
        inchi, c = drugCountSorted[i]
        if c < MIN_D:
            continue
        validInchi.add(inchi)

    dADR2Pair = dict()
    # Filter by ADRs
    allPairs = set()
    s1 = 0
    for pairs, adrs in drugPair2ADR.items():
        inchi1, inchi2 = pairs
        if inchi1 in validInchi and inchi2 in validInchi:
            for adr in adrs:
                if adr in validADRs:
                    pairs = utils.get_insert_key_dict(dADR2Pair, adr, [])
                    pairs.append((inchi1, inchi2))
                    allPairs.add((inchi1, inchi2))
                    s1 += 1
    print_db("Saving ", s1)
    print_db(len(allPairs), len(drugPair2ADR), len(dADR2Pair))
    v = (params.MAX_R_ADR, params.MAX_R_DRUG, dADR2Pair, orderedADR, inchi2FingerPrint)
    utils.save_obj(v, DUMP_FILE)
    return v


def swap(d1, d2):
    if d1 > d2:
        d1, d2 = d2, d1
    return d1, d2


def genTrueNegTpl(adrId2Pairid, nDrug, nNegPerADR, kSpace=params.KSPACE):
    negTpls = []

    allPairs = set()
    for pairs in adrId2Pairid.values():
        for pair in pairs:
            allPairs.add(pair)

    for adrId, pairs in adrId2Pairid.items():
        adrId = adrId + nDrug
        ni = 0

        if kSpace:
            for pair in allPairs:
                d1, d2 = pair

                d1, d2 = swap(d1, d2)
                p = (d1, d2)
                if p not in pairs:
                    negTpls.append((d1, d2, adrId))
            continue

        nx = nNegPerADR * np.log(10) / (np.log(len(pairs) + 3))

        while ni < nx:
            d1, d2 = np.random.randint(0, nDrug, 2)
            d1, d2 = swap(d1, d2)
            pair = (d1, d2)
            if pair not in pairs:
                ni += 1
                negTpls.append((d1, d2, adrId))
    return negTpls


def getPairTypeById(id1, id2, anchor):
    # Drug-Drug: 0, Drug-Side effect: 1, Side effect - Drug: 2, Drug-self-loop: 3, Side effect self loop: 4
    tp = 0
    if id1 < anchor <= id2:
        tp = 1
    elif id1 >= anchor > id2:
        tp = 2
    elif id1 == id2:
        if id1 < anchor:
            tp = 3
        else:
            tp = 4
    return tp


def producer(data):
    dADRId2PairIds, numDrug, numNodes, iFold, numSe, negFold, features, smiles, edgeIndex, nProtein, orderedADRIds = data
    testFold = []
    trainFold = []
    validFold = []

    edgeSet = set()
    edge2Label = dict()

    for adr, pairs in dADRId2PairIds.items():

        adr = adr + numDrug
        pairs = sorted(list(pairs))
        pairs = copy.deepcopy(pairs)
        random.seed(params.TORCH_SEED)
        random.shuffle(pairs)
        nSize = len(pairs)
        foldSize = int(nSize / params.K_FOLD)
        startTest = iFold * foldSize
        endTest = (iFold + 1) * foldSize
        if endTest > nSize:
            endTest = nSize

        if iFold == params.K_FOLD - 1:
            startValid = 0
        else:
            startValid = endTest

        endValid = startValid + foldSize

        for i in range(nSize):
            d1, d2 = pairs[i]
            tpl = (d1, d2, adr)
            if params.HIGH_TWOSIDES:
                trainFold.append(tpl)
                edgeSet.add((d1, d2))
                labels = utils.get_insert_key_dict(edge2Label, (d1, d2), [])
                labels.append(adr - numDrug)

            if startTest <= i < endTest:
                testFold.append(tpl)
            elif startValid <= i < endValid:
                validFold.append(tpl)

            elif not params.HIGH_TWOSIDES:
                trainFold.append(tpl)
                edgeSet.add((d1, d2))
                labels = utils.get_insert_key_dict(edge2Label, (d1, d2), [])
                labels.append(adr - numDrug)

    pairStats = trainFold2PairStats(trainFold, numDrug)

    testPosPair2Label = dict()
    validPosPair2Label = dict()
    testNegPair2Label = dict()

    for tpl in testFold:
        d1, d2, adr = tpl
        posLabels = utils.get_insert_key_dict(testPosPair2Label, (d1, d2), [])
        posLabels.append(adr - numDrug)

    for tpl in validFold:
        d1, d2, adr = tpl
        posLabels = utils.get_insert_key_dict(validPosPair2Label, (d1, d2), [])
        posLabels.append(adr - numDrug)

    for tpl in negFold:
        d1, d2, adr = tpl
        negLabels = utils.get_insert_key_dict(testNegPair2Label, (d1, d2), [])
        negLabels.append(adr - numDrug)

    for edge in edgeSet:
        d1, d2 = edge
        edgeIndex.append([d1, d2])
        edgeIndex.append([d2, d1])

    def getPairTypeById(id1, id2, anchor):
        # Drug-Drug: 0, Drug-Side effect: 1, Side effect - Drug: 2, Drug-self-loop: 3, Side effect self loop: 4
        tp = 0
        if id1 < anchor <= id2:
            tp = 1
        elif id1 >= anchor > id2:
            tp = 2
        elif id1 == id2:
            if id1 < anchor:
                tp = 3
            else:
                tp = 4
        return tp

    hyperEdgeCliqueIndex = []
    hyperedgeIndexType = []
    for tpl in trainFold:
        for pair in list(itertools.product(tpl, tpl)):
            id1, id2 = pair
            if id1 != id2:
                tp = getPairTypeById(id1, id2, numDrug)
                hyperEdgeCliqueIndex.append([id1, id2])
                hyperedgeIndexType.append(tp)

    # Adding self-loop
    for idx in range(numDrug + numSe):
        id1 = idx
        tp = getPairTypeById(id1, id1, numDrug)
        hyperEdgeCliqueIndex.append([id1, id1])
        hyperedgeIndexType.append(tp)

    realFold = RealFoldData(trainFold, testFold, validFold, 0, 0, negFold, features)
    realFold.nSe = numSe
    realFold.nD = numDrug
    realFold.DFold = 0

    realFold.trainPairStats = pairStats
    realFold.iFold = iFold

    realFold.pEdgeSet = edgeSet
    realFold.pTrainPair2Label = edge2Label
    realFold.pValidPosLabel = validPosPair2Label
    realFold.pTestPosLabel = testPosPair2Label
    realFold.pTestNegLabel = testNegPair2Label
    realFold.dADR2Drug = dADRId2PairIds
    realFold.batchSMILE = smiles
    realFold.ppGraph = edgeIndex
    realFold.nPro = nProtein
    realFold.orderADRIds = orderedADRIds
    realFold.hyperEdgeCliqueIndex = torch.tensor(np.asarray(hyperEdgeCliqueIndex), dtype=torch.long).t().contiguous()
    realFold.hyperEdgeIndexType = np.asarray(hyperedgeIndexType, dtype=int)

    return realFold


def genBatchAtomGraph(smiles):
    moleculeFactory = MoleculeFactory()
    for smile in smiles:
        moleculeFactory.addSMILE(smile)
    graphBatch = moleculeFactory.createBatchGraph(atomOffset=0)
    return graphBatch


def genSMILESFromInchies(inchies):
    inchi2SMILE = loadInchi2SMILE()
    allSMILEs = []
    for inchi in inchies:
        smile = inchi2SMILE[inchi]
        allSMILEs.append(smile)
    return allSMILEs


def genHyperData(onlyFirst=False):
    nADR, nDrug, dADR2Pair, orderedADR, inchi2FingerPrint = utils.load_obj(DUMP_FILE)
    print_db(nADR, len(dADR2Pair), nDrug, len(inchi2FingerPrint))

    dADR2Id = dict()
    dInchi2Id = dict()
    dADRId2PairIds = dict()

    adrs = sorted(list(dADR2Pair.keys()))
    allPairs = set()
    orderedADRIds = list()
    for adr in adrs:
        adrId = utils.get_update_dict_index(dADR2Id, adr)
        pairs = dADR2Pair[adr]
        for pair in pairs:
            inchi1, inchi2 = pair
            d1 = utils.get_update_dict_index(dInchi2Id, inchi1)
            d2 = utils.get_update_dict_index(dInchi2Id, inchi2)
            d1, d2 = swap(d1, d2)
            pairIds = utils.get_insert_key_dict(dADRId2PairIds, adrId, set())
            pairIds.add((d1, d2))
            allPairs.add((d1, d2))
    for oADr in orderedADR:
        adrId = dADR2Id[oADr]
        orderedADRIds.append(adrId)
    print_db("Drug, ADR, Pairs: ", len(dInchi2Id), len(adrs), len(allPairs))
    print_db("Loading ADR 2 Pair completed")

    numDrug = len(dInchi2Id)
    numSe = len(dADR2Id)
    numNodes = numDrug + numSe
    print_db(numDrug, numSe, numNodes)
    # Create Feature Matrix:

    dDrugId2Inchi = utils.reverse_dict(dInchi2Id)
    allInchies = dDrugId2Inchi.keys()

    features = []
    inchies = []
    for i in range(numDrug):
        inchi = dDrugId2Inchi[i]
        inchies.append(inchi)
        fs = inchi2FingerPrint[inchi]
        features.append(fs)

    smiles = genSMILESFromInchies(inchies)

    edgeIndex, protein2Id, nDrug, dDrug2ProteinFeatures = loadDrug2Protein(inchies, params.PATHWAY)
    if params.HIGH_TWOSIDES:
        utils.save_obj(protein2Id, "%s/TWOSIDESProtein2Id_1.dat" % DATASET_DIR)

    appendProteinProtein(protein2Id, edgeIndex, nDrug)

    nProtein = len(protein2Id)
    features = np.vstack(features)

    if params.PROTEIN_FEATURE:
        pFeatures = []
        cc = 0
        for inchi in inchies:
            try:
                ff = dDrug2ProteinFeatures[inchi]
            except:
                ff = np.zeros(nProtein)
                print("No Protein for: ", inchi)
                cc += 1
            pFeatures.append(ff)
        print_db("Missing: ", cc)
        pFeatures = np.vstack(pFeatures)
        features = np.concatenate((features, pFeatures), axis=1)
    if params.ONE_HOT:
        nD = features.shape[0]
        features = np.diag(np.ones(nD))
    print_db("Feature: ", features.shape)
    if params.HIGH_TWOSIDES:
        utils.save_obj((protein2Id, features), "%s/TWOSIDESfeatures_1.dat" % DATASET_DIR)

    negFold = genTrueNegTpl(dADRId2PairIds, numDrug, params.SAMPLE_NEG)
    print("Starting...")
    for iFold in range(params.K_FOLD):
        print("Gen fold: ", iFold)
        edgeIndex2 = copy.deepcopy(edgeIndex)
        data = dADRId2PairIds, numDrug, numNodes, iFold, numSe, negFold, features, smiles, edgeIndex2, nProtein, orderedADRIds
        realFold = producer(data)
        print("Saving fold: ", iFold)
        pref = ""
        if params.HIGH_TWOSIDES:
            pref = "fullC5"
        utils.save_obj(realFold, "%s/%s_%d_%d_%d_%d" % (
            DATASET_DIR, pref, params.MAX_R_ADR, params.MAX_R_DRUG, params.ADR_OFFSET, iFold))
        if onlyFirst:
            break


def trainFold2PairStats(trainFold, nOffDrug):
    dii = dict()
    dij = dict()
    dit = dict()
    dtt = dict()
    for tpl in trainFold:
        i, j, t = tpl
        to = t - nOffDrug
        i, j = swap(i, j)

        vdii = utils.get_insert_key_dict(dii, (i, i), [])
        vdii.append(to)
        vdjj = utils.get_insert_key_dict(dii, (j, j), [])
        vdjj.append(to)

        vdji = utils.get_insert_key_dict(dij, (i, j), [])
        vdji.append(to)

        vdtt = utils.get_insert_key_dict(dtt, (t, t), [])
        vdtt.append(to)

        vdit = utils.get_insert_key_dict(dit, (i, t), [])
        vdit.append(to)
        vdjt = utils.get_insert_key_dict(dit, (j, t), [])
        vdjt.append(to)

    def dict2Array(d):
        d2 = dict()
        for k, v in d.items():
            v = np.asarray(v, dtype=int)
            d2[k] = v
        return d2

    return dict2Array(dii), dict2Array(dij), dict2Array(dit), dict2Array(dtt)


def saveId2Name(inp=params.PATH_TWOSIDES_C5):
    print_db("DRUG, ADR: ", params.MAX_R_DRUG, params.MAX_R_ADR)
    createSubSet(inp)
    nADR, nDrug, dADR2Pair, orderedADR, inchi2FingerPrint = utils.load_obj(DUMP_FILE)
    print_db(nADR, len(dADR2Pair), nDrug, len(inchi2FingerPrint))

    # Convert 2 Id
    dADR2Id = dict()
    dInchi2Id = dict()
    dADRId2PairIds = dict()

    adrs = sorted(list(dADR2Pair.keys()))
    allPairs = set()
    orderedADRIds = list()
    for adr in adrs:
        adrId = utils.get_update_dict_index(dADR2Id, adr)
        pairs = dADR2Pair[adr]
        for pair in pairs:
            inchi1, inchi2 = pair
            d1 = utils.get_update_dict_index(dInchi2Id, inchi1)
            d2 = utils.get_update_dict_index(dInchi2Id, inchi2)
            d1, d2 = swap(d1, d2)
            pairIds = utils.get_insert_key_dict(dADRId2PairIds, adrId, set())
            pairIds.add((d1, d2))
            allPairs.add((d1, d2))

    id2ADr = utils.reverse_dict(dADR2Id)
    id2Inchi = utils.reverse_dict(dInchi2Id)
    fin = open(params.DRUGBANK_ATC_INCHI)
    dINCHI2Name = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("\t")
        dINCHI2Name[parts[-1]] = parts[1]
    fin.close()
    dId2DrugName = dict()
    for i in range(len(id2Inchi)):
        dId2DrugName[i] = dINCHI2Name[id2Inchi[i]]
    utils.save_obj((id2ADr, dId2DrugName), params.ID2NamePath_TWOSIDEC5)


def getBackId(s, d1x, d2x, db=False):
    id2ADr, dId2DrugName = utils.load_obj(params.ID2NamePath_TWOSIDEC5)
    if db:
        print("S_", s, id2ADr[s])
        print("D_", d1x, dId2DrugName[d1x])
        print("D_", d2x, dId2DrugName[d2x])
    return id2ADr[s], dId2DrugName[d1x], dId2DrugName[d2x]


def exportData():
    params.ON_REAL = True
    params.DEG_NORM = True
    print_db("DRUG, ADR: ", params.MAX_R_DRUG, params.MAX_R_ADR)

    createSubSet()
    genHyperData()


def exportFullTWOSIDEs():
    params.ON_REAL = True
    params.DEG_NORM = True
    print_db("DRUG, ADR: ", params.MAX_R_DRUG, params.MAX_R_ADR)
    createSubSet(params.PATH_TWOSIDES_C5)
    genHyperData(onlyFirst=True)


def writeSED():
    id2ADr, dId2DrugName = utils.load_obj(params.ID2NamePath_TWOSIDEC5)
    fSE = open("%s/SeIdNameC5.txt" % DATASET_DIR, "w")
    for k in range(len(id2ADr)):
        v = id2ADr[k]
        fSE.write("%s\t%s\n" % (k, v))
    fSE.close()

    fDrug = open("%s/DrugId2NameC5.txt" % DATASET_DIR, "w")
    for k in range(len(dId2DrugName)):
        v = dId2DrugName[k]
        fDrug.write("%s\t%s\n" % (k, v))
    fDrug.close()


def run():
    # For TWOSIDES, there are two data files: one with high quality for only high confident interactions for full
    # training and extraction (TWOSIDES_C5) and one for all interactions for running K-Folds (TWOSIDE_ALL)

    print("High quality TWOSIDES for full training: ", params.HIGH_TWOSIDES)
    if params.HIGH_TWOSIDES:
        global DUMP_FILE
        DUMP_FILE = "%s_fullC5" % DUMP_FILE
        resetRandomSeed()
        saveId2Name()
        writeSED()
        resetRandomSeed()
        exportFullTWOSIDEs()
    else:
        exportData()


if __name__ == "__main__":
    run()
