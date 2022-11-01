import dataFactory.dataLoader
import params
from utils import utils
import random
import copy
from dataFactory.moleculeFactory import MoleculeFactory

from multiprocessing import Process, Value, Queue

from dataFactory.dataLoader import RealFoldData
import itertools
import time
import numpy as np
import torch

# Set the size of holdout negative samples for each pair of drugs (Used for calculating AUC, AUPR)
# This number is depended on the number of positive samples for each drug pair of the dataset
# The number is selected that the reported results of baseline methods (e.g. DECAGON) can be reproduced
# All methods (containing baselines) used the same training, testing data (and holdout negative samples)
# of each dataset
params.SAMPLE_NEG = 1300

# Set dataset DIR
DATASET_DIR = "%s/TWOSIDES" % params.TMP_DIR
utils.ensure_dir(DATASET_DIR)

# Temporal file for processing the dataset
DUMP_FILE = "%s/dump.pkl" % DATASET_DIR


def print_db(*msg):
    r"""
    Debugging with PRINT_DB flag
    Args:
        *msg: debugging messages

    """
    if params.PRINT_DB:
        print(*msg)


def resetRandomSeed():
    r"""
    Set random seed for random generator

    """
    import random
    random.seed(params.TORCH_SEED)
    torch.manual_seed(params.TORCH_SEED)
    np.random.seed(params.TORCH_SEED)


def loadPubChem():
    r"""
        # Load a dictionary from drug inchikey to corresponding 881 pubchem substructure fingerprint
    Returns:
        Return dictionary {druginchikey : np.array(881)}
    """

    return utils.load_obj(params.PUBCHEM_FILE)


def loadMonoADR():
    r"""
    # Load Side effect of single drugs
    # According to Decagon paper (https://academic.oup.com/bioinformatics/article/34/13/i457/5045770)
    # The side effects of single drug will be excluded from the side effects of drug-drug interactions.
    Input file:
    params.MONO_ADR_FILE
    Format of each line:
    drugName|Inchikey|side_effect_1,side_effect_2,....
    Returns:
    dictionary {drugInchi : [list of mono adr]}
    """

    fin = open(params.MONO_ADR_FILE)

    # Dictionary fro drugInchi to list of adrs
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
    r"""
    # Load mapping from protein uniprot Id to KEGG pathways in KEGG dataset
    # Format of the input file: each line in uniprot_2_pathway.txt
    # ProteinUniProtID|pathway_1,pathway_2,..., pathway_2
    Returns:
        dProtein2Pathways: dictionary {proteinUniProtId: [pathway_interger_id_1, pathway_interger_id_2,...]
        dPathway2Id: dictionary {pathway_kegg_id: pathway_integer_id}
        dPathway2Name: dictionary {pathay_kegg_id : pathway_long_name_description}

    """
    fin = open("%s/KEGG/uniprot_2_pathway.txt" % params.DATA_DIR)
    # Dictionary for pathway_kegg_id to pathway_integer_id
    dPathway2Id = dict()
    # Dictionary for each protein_uniprot_id to a list of pathway_integer_ids
    dProtein2Pathways = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("|")
        # Protein Uniprot id
        protein = parts[0]
        # LIst of pathway kegg ids
        pathways = parts[1].split(",")
        # List of corresponding pathway integer ids
        # Read comments in utils.get_update_dict_index for detail
        ar = [utils.get_update_dict_index(dPathway2Id, pathway) for pathway in pathways]
        # Assign protein uniprot id with the corresponding array of pathway_integer_ids
        dProtein2Pathways[protein] = ar
    fin.close()

    # Dictionary for pathway_kegg_id to pathwah long name description
    # Used for post-processing and intepratation
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
    r"""
    Load proteins (pathways) of given drug inchikeys
    Args:
        inchies: list of drug inchikeys
        pathway: Flag for using pathway information or not

    Returns:
         edge_index: List of pairs of [(drug_index, protein_index),....], used to construct edges of drug-protein
         protein2Id: dictionary of {protein_uniprot_id: protein_integer_id}
         nDrug: Number of drug
         dDrug2ProteinFeatures: dictionary {drug_inchikey: binary_array_of_corresponding_protein_ids]


    """
    # Dictionary from drug inchikey to drug integer ids
    dInchi2Id = dict()
    # Assign index to drug id given drug inchikey with utils.get_update_dict_index function
    for inchi in inchies:
        utils.get_update_dict_index(dInchi2Id, inchi)
    nDrug = len(dInchi2Id)
    # Load map from drug inchikey to protein uniprot list
    drug2ProteinList = dataFactory.dataLoader.loadDrugProteinMap()

    proteinListList = sorted(list(drug2ProteinList.values()))

    # Set of protein uniprot ids
    protensSets = set()
    # Dictionary from protein uniprot to integer id
    protein2Id = dict()
    # Load associated pathways of proteins
    dProtein2Pathways, dPathway2Id, _ = loadProtein2Pathway()

    # Get all protein set (all proteins)
    for proteins in proteinListList:
        for protein in proteins:
            if protein != "":
                protensSets.add(protein)

    proteinList = list(protensSets)
    # Sort protein by uniprot id to have a persistant order
    proteinList = sorted(proteinList)
    # Assign protein id to protein uniprot id

    for protein in proteinList:
        utils.get_update_dict_index(protein2Id, protein)

    # Dictionary for drug inchikey to corresponding proteins
    dDrug2ProteinFeatures = dict()
    # Number of proteins
    nProteins = len(protein2Id)
    # Number of pathways
    nPathways = len(dPathway2Id)
    # Feature size
    # Init by the number of protein
    nS = nProteins
    # If use pathway as features, then add with the number of pathways
    if pathway:
        nS += nPathways
    # Store pair of drug-protein indices
    edge_index = []

    cc = 0
    # Assign drug features by iterating drug-protein map
    for drugInchi, proteins in drug2ProteinList.items():
        drugId = utils.get_dict(dInchi2Id, drugInchi, -1)
        # Only select drug appear on the input inchikeys
        if drugId == -1:
            cc += 1
            continue
        # Init drug features by zeros
        proteinFeature = np.zeros(nS)
        # Assign 1 values to drug features of the corresponding protein indices and pathway indices
        for p in proteins:
            if p == "":
                continue
            piD0 = protein2Id[p]
            proteinFeature[piD0] = 1
            if pathway:
                pa = utils.get_dict(dProtein2Pathways, p, [])
                for a in pa:
                    proteinFeature[a + nProteins] = 1

            pId = piD0 + nDrug
            # Add pair of drug-protein as an edge
            edge_index.append([drugId, pId])
            # Consider two directions
            edge_index.append([pId, drugId])
        dDrug2ProteinFeatures[drugInchi] = proteinFeature
    return edge_index, protein2Id, nDrug, dDrug2ProteinFeatures


def appendProteinProtein(protein2Id, edg_index, nDrug):
    r"""
    In cases of using protein-protein interactions,
    add new protein-protein edges to the edge list of drug-protein
    Args:
        protein2Id: dictionary from protein uniprot to integer id
        edg_index: [list of pair of indices of drug-protein(protein-drug)]
        nDrug: number of drug

    Returns: new edge_list after adding

    """
    # Open protein-protein interaction file
    fin = open(params.PPI_FILE)
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        # Get corresponding protein ids from protein uniprot id
        p1 = utils.get_dict(protein2Id, parts[0], -1)
        p2 = utils.get_dict(protein2Id, parts[1], -1)
        # If both proteins have corresponding ids, then add new edge
        if p1 != -1 and p2 != -1:
            edg_index.append([p1 + nDrug, p2 + nDrug])
            edg_index.append([p2 + nDrug, p1 + nDrug])

    fin.close()
    return edg_index


def loadInchi2SMILE():
    r"""
    Load drug inchikey to SMILE representation
    For SMILE representation: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system

    Returns:
        dictionary : {drug_inchikey: SMILE}

    """
    # Open data file for drug inchikey - SMILE
    f = open(params.DRUGBANK_ATC_INCHI)
    # Dictionary for drug_inchi to SMILE
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
    r"""
    Creat a subset of DDI data for training after filtering invalid names
    or drugs (side effects) with two small number of samples

    Args:
        inp_path: DDI data file. Format of each line:
        drugName1|drugName2|drugInchikey1|drugInchikey2|side_effect_1,side_effect_2,....



    """

    def loadInvalidSes():
        # Load invalid tokens for side effects
        return set(line.strip() for line in open("%s/exception_adrs.txt" % params.DDI_DIR).readlines())

    # Load map from Inchikey to Chemical fingerprint of 881 Pubchem substructures
    inchi2FingerPrint = loadPubChem()
    # Set of inchikeys
    inchiKeys = inchi2FingerPrint.keys()
    # Side effect list of single drug
    monoADR = loadMonoADR()

    # Open input file
    fin = open(inp_path)
    # Set of drug
    drugSet = set()
    # Set of side effects (ADRs)
    adrSet = set()
    # Dictionary for counter of drugs (drug frequencies)
    drugCount = dict()
    # Dictionary of counter of side effect (side effect frequency)
    adrCount = dict()
    # Pair of drug to corresponding side effects
    drugPair2ADR = dict()
    # Map from inchikey to drug names
    inchi2Drug = dict()

    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("|")
        # Split to drug name, inchikeys
        d1, d2, inchi1, inchi2 = parts[0], parts[1], parts[2], parts[3]
        if inchi1 not in inchiKeys or inchi2 not in inchiKeys:
            continue
        # List of adrs
        adrs = parts[4].split(",")
        # Assign map from inchikey to drug name
        inchi2Drug[inchi1] = d1
        inchi2Drug[inchi2] = d2
        # Add drug to drug set
        drugSet.add(inchi1)
        drugSet.add(inchi2)
        # Add frequency for drug
        utils.add_dict_counter(drugCount, inchi1)
        utils.add_dict_counter(drugCount, inchi2)
        #
        # utils.get_dict(monoADR, inchi1, set())
        # utils.get_dict(monoADR, inchi2, set())
        # Add adr to set and frequencies
        for adr in adrs:
            adrSet.add(adr)
            utils.add_dict_counter(adrCount, adr)
        # Assign drug inchi pair to corresponding adrs
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
    # Filter adr list not containing invalid tokens
    # Keep the order of adr
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

    # Filter drug by min frequencies
    drugCountSorted = utils.sort_dict(drugCount)
    validInchi = set()
    m = min(len(drugCount), params.MAX_R_DRUG)
    MIN_D = 5
    for i in range(m):
        inchi, c = drugCountSorted[i]
        if c < MIN_D:
            continue
        validInchi.add(inchi)

    dADR2Pair = dict()

    # Create mapping from ach ADR to a list of pairs of drug inchikeys
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
    # Save temporal file for number MAX_ADR, MAX_DRUG, mapping from ADR name to list of pair of drug inchikey, list of ordered ADR names,
    # map from drug inchikey to fingerprint
    v = (params.MAX_R_ADR, params.MAX_R_DRUG, dADR2Pair, orderedADR, inchi2FingerPrint)
    utils.save_obj(v, DUMP_FILE)
    return v


def swap(d1, d2):
    r"""
    Swap drug by the name (id) order
    Args:
        d1: drug1
        d2: drug2

    Returns:
        ordered drugs
    """
    if d1 > d2:
        d1, d2 = d2, d1
    return d1, d2


def genTrueNegTpl(adrId2Pairid, nDrug, nNegPerADR, kSpace=params.KSPACE):
    r"""
    Sample holdout true negative triples of drug-drug-side effect that does not appear on the input DDIs
    The holdout true negative set is used for calculating AUC, AUPR when comparing the methods
    the same holdout true negative is used for all methods when comparing performance.
    Args:
        adrId2Pairid: Dictionary for ADR integerId to pair of drug id
        nDrug: number of side effect
        nNegPerADR: number of negative sample for each ADR (which is normally inversed proprotional to the length of the input)
        kSpace: If the negative pair is only on the given pair of the training (by default, this option is false)

    Returns:
        list of holdout true negative samples [(drug1, drug2, side_effect),...]
    """
    # List of holdout true negative samples
    negTpls = []
    # Set of drug pair
    allPairs = set()
    # Get all drug pair
    for pairs in adrId2Pairid.values():
        for pair in pairs:
            allPairs.add(pair)
    # Sample holdout true negative samples for each ADR
    for adrId, pairs in adrId2Pairid.items():
        adrId = adrId + nDrug
        ni = 0
        # In cases of kSpace, only sample on the given drug pairs (Skip by default)
        if kSpace:
            for pair in allPairs:
                d1, d2 = pair

                d1, d2 = swap(d1, d2)
                p = (d1, d2)
                if p not in pairs:
                    negTpls.append((d1, d2, adrId))
            continue
        # Get the length of the negative samples: inversed proprotional to the length of the input
        # Assuming that the number of the negative samples for an ADR with fewer interactions is larger.
        nx = nNegPerADR * np.log(10) / (np.log(len(pairs) + 3))
        # Sample nx holdout negative samples that not appear on the coresponding input drug pair of the adr
        while ni < nx:
            d1, d2 = np.random.randint(0, nDrug, 2)
            d1, d2 = swap(d1, d2)
            pair = (d1, d2)
            if pair not in pairs:
                ni += 1
                negTpls.append((d1, d2, adrId))
    return negTpls


def getPairTypeById(id1, id2, anchor):
    r"""
    Given a pair of ids, return type of the pair
    Args:
        id1: Id of the first element
        id2: id of the second element
        anchor: max_drug_id

    Returns:
        Drug-Drug: 0, Drug-Side effect: 1, Side effect - Drug: 2, Drug-self-loop: 3, Side effect self loop: 4

    """
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
    r"""
        A function used for processing a fold data:
    Args:
        data: input fold data

    Returns:
        processed input fold data with train-test-validation split for hyperedges.
        Data is stored in a dataFactory.dataLoader.RealFoldData instance

    """
    # Decompose data for the fold i in k-Fold
    dADRId2PairIds, numDrug, numNodes, iFold, numSe, negFold, features, smiles, edgeIndex, nProtein, orderedADRIds = data
    # List of triples of drug-drug-side effects ( Hyperedges)
    testFold = []
    trainFold = []
    validFold = []

    # Set of edges
    edgeSet = set()
    # Edge to labels
    edge2Label = dict()
    # Assign hyperedges (triples) to train-test-validation
    for adr, pairs in dADRId2PairIds.items():
        # pairs of drug ids of given adr
        # Assign adr id by adding the number of drug
        # (e.g. Given 100 drug, then the id of drugs: 0,...99, the id of ADR become: 100, 101, ...

        adr = adr + numDrug
        # Deep copy list the pair (Since we will modify the (order) of the list)
        # By using deep copy, we can use for multiprocessing
        pairs = sorted(list(pairs))
        pairs = copy.deepcopy(pairs)
        # Fix seed value for shuffling
        random.seed(params.TORCH_SEED)
        random.shuffle(pairs)
        # Number of pairs
        nSize = len(pairs)
        # Number of hyperedges (triples) for each fold
        foldSize = int(nSize / params.K_FOLD)
        # Starting position of the triple for testing
        startTest = iFold * foldSize
        # Ending position of the triple for testing
        endTest = (iFold + 1) * foldSize
        # The ending position can not exceed nSize
        if endTest > nSize:
            endTest = nSize
        # Start and Ending positions for validations
        if iFold == params.K_FOLD - 1:
            startValid = 0
        else:
            startValid = endTest

        endValid = startValid + foldSize

        # Assigning the pair to train-test-validation
        # Checking for TWOSIDE Highquality that in that case, all hypereges are on the training data.

        for i in range(nSize):
            d1, d2 = pairs[i]
            # Creat hypepedge (triples of drug-drug-side effect)
            tpl = (d1, d2, adr)
            if params.HIGH_TWOSIDES:
                # For training full high quality TWOSIDES, add all triples to training data
                trainFold.append(tpl)
                # Add a new edge
                edgeSet.add((d1, d2))
                # Get the label array of the corresponding edge
                labels = utils.get_insert_key_dict(edge2Label, (d1, d2), [])
                # Reassign the Id of adr for the edge that the label of edge (side effect) starts from 0 ( for adr 0)
                labels.append(adr - numDrug)
            # Check where the current hypedges is on train-test-validation
            if startTest <= i < endTest:
                testFold.append(tpl)
            elif startValid <= i < endValid:
                validFold.append(tpl)

            elif not params.HIGH_TWOSIDES:
                trainFold.append(tpl)
                edgeSet.add((d1, d2))
                labels = utils.get_insert_key_dict(edge2Label, (d1, d2), [])
                labels.append(adr - numDrug)
    # Get statistics of the drug-pairs in the training data
    pairStats = trainFold2PairStats(trainFold, numDrug)

    # Dictionary for pair of drug to corresponding positive labels
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

    # Create clique expansion for the hypeperedge on training data
    # e.g. (d1, d2, e) -> (d1, d2), (d1, e), (d2, e)
    hyperEdgeCliqueIndex = []
    hyperedgeIndexType = []
    for tpl in trainFold:
        for pair in list(itertools.product(tpl, tpl)):
            id1, id2 = pair
            if id1 != id2:
                tp = getPairTypeById(id1, id2, numDrug)
                hyperEdgeCliqueIndex.append([id1, id2])
                hyperedgeIndexType.append(tp)

    # Adding self-loop : drug node to itself, side efffect node to itself
    for idx in range(numDrug + numSe):
        id1 = idx
        tp = getPairTypeById(id1, id1, numDrug)
        hyperEdgeCliqueIndex.append([id1, id1])
        hyperedgeIndexType.append(tp)

    # Create dataFactory.dataLoader.RealFoldData instance to store the information of the fold
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
    r"""
    Generating a graph of atoms from each drug SMILE representation
    Convert into a graph bath

    This function is used to generate data for atom based models (baseline methods, not for SPARSE)

    Args:
        smiles: list of SMILES

    Returns: batch of graph

    """
    moleculeFactory = MoleculeFactory()
    for smile in smiles:
        moleculeFactory.addSMILE(smile)
    graphBatch = moleculeFactory.createBatchGraph(atomOffset=0)
    return graphBatch


def genSMILESFromInchies(inchies):
    r"""
    Load dictionary for SMILES of each drug inchikey
    Args:
        inchies: list of drug inchikeys

    Returns:
        corresponding drug SMILES
    """
    inchi2SMILE = loadInchi2SMILE()
    allSMILEs = []
    for inchi in inchies:
        smile = inchi2SMILE[inchi]
        allSMILEs.append(smile)
    return allSMILEs


def genHyperData(onlyFirst=False):
    r"""
        Generating hypergraph data for all folds
    Args:
        onlyFirst: Only generate the first fold (fold-0, used for debugging). By default: False


    """
    # Load dump data from corresponding subset of the DDI data
    nADR, nDrug, dADR2Pair, orderedADR, inchi2FingerPrint = utils.load_obj(DUMP_FILE)
    print_db(nADR, len(dADR2Pair), nDrug, len(inchi2FingerPrint))

    dADR2Id = dict()
    dInchi2Id = dict()
    dADRId2PairIds = dict()

    adrs = sorted(list(dADR2Pair.keys()))
    allPairs = set()
    orderedADRIds = list()
    # Generating dictionary of adr integer id to list of pairs of drug ids
    # from dictionary of adr name to list pairs of drug inchikeys
    for adr in adrs:
        # Get adr integer id (index) from its name
        adrId = utils.get_update_dict_index(dADR2Id, adr)
        # Get corresponding list of drug inchikeys of the adr
        pairs = dADR2Pair[adr]
        # Convert pair of drug inchikeys to pair of drug integer ids
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

    # Create Drug Feature Matrix, row ith correspond to drug id i

    dDrugId2Inchi = utils.reverse_dict(dInchi2Id)
    allInchies = dDrugId2Inchi.keys()

    features = []
    inchies = []
    for i in range(numDrug):
        inchi = dDrugId2Inchi[i]
        inchies.append(inchi)
        fs = inchi2FingerPrint[inchi]
        features.append(fs)

    # SMILES of given drug inchikeys
    smiles = genSMILESFromInchies(inchies)

    # Load drug-protein association
    edgeIndex, protein2Id, nDrug, dDrug2ProteinFeatures = loadDrug2Protein(inchies, params.PATHWAY)

    # If it is HIGH quality TWOSIDES (TWOSIDE_C5),
    # then save the mapping from protein name to id for later intepretation
    if params.HIGH_TWOSIDES:
        utils.save_obj(protein2Id, "%s/TWOSIDESProtein2Id_1.dat" % DATASET_DIR)

    # Append protein-protein interactions
    appendProteinProtein(protein2Id, edgeIndex, nDrug)

    # Create feature matrix
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
    # In a case of using ONE_HOT feature vector for drugs
    if params.ONE_HOT:
        nD = features.shape[0]
        features = np.diag(np.ones(nD))
    print_db("Feature: ", features.shape)
    # In case of HIGH quality TWOSIDES, save feature matrix
    if params.HIGH_TWOSIDES:
        utils.save_obj((protein2Id, features), "%s/TWOSIDESfeatures_1.dat" % DATASET_DIR)
    # Generating holdout true negative hyperedges
    negFold = genTrueNegTpl(dADRId2PairIds, numDrug, params.SAMPLE_NEG)
    print("Starting...")
    # Generating fold data
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
    r"""
    Do statistics for pairs of drug ids
    Used for input data of CentSmoothie model.
    (Read CentSmoothie paper for detail https://arxiv.org/abs/2112.07837).
    (For SPARSE only, you can skip this function)
    Args:
        trainFold: Train hyperedges
        nOffDrug: number of drugs

    Returns: Count for number of occurences of drugs, side effects

    """
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
    r"""
    Save mapping from integer id to string name of drugs, side effects, proteins, pathways
    Used for post processing and interpretation

    Args:
        inp: path to high-quality data

    Returns:

    """

    # Create subset then sort the names to create mapping dictionary
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
    # Create mapping from ADR integer Id to list of pairs of drug integer id
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

    # Reverse dictionaries of name_to_id to id_to_name
    id2ADr = utils.reverse_dict(dADR2Id)
    id2Inchi = utils.reverse_dict(dInchi2Id)

    # Get drug inchikey to name mapping
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
    r"""
        # Get names of side effect, drugs from ids

    Args:
        s: side effect (integer) id
        d1x: drug1 id
        d2x: drug2 id
        db: debug flag

    Returns:
        side effect name, drug1Name, drug2Name
    """
    id2ADr, dId2DrugName = utils.load_obj(params.ID2NamePath_TWOSIDEC5)
    if db:
        print("S_", s, id2ADr[s])
        print("D_", d1x, dId2DrugName[d1x])
        print("D_", d2x, dId2DrugName[d2x])
    return id2ADr[s], dId2DrugName[d1x], dId2DrugName[d2x]


def exportData():
    r"""
    Export data by creating a subset (after filtering) then generating hypergraph data

    """
    params.ON_REAL = True
    params.DEG_NORM = True
    print_db("DRUG, ADR: ", params.MAX_R_DRUG, params.MAX_R_ADR)

    # Create subset of filtered data
    createSubSet()
    # Generating hypergraph data
    genHyperData()


def exportFullTWOSIDEs():
    r"""
    Exporting high quality TWOSIDES (TWOSIDE_C5) for full training data

    """
    # Set Flag for real data
    params.ON_REAL = True
    # Set Flag for normalizing node degree
    params.DEG_NORM = True
    print_db("DRUG, ADR: ", params.MAX_R_DRUG, params.MAX_R_ADR)
    createSubSet(params.PATH_TWOSIDES_C5)
    genHyperData(onlyFirst=True)


def writeSED():
    r"""
    Save mapping from SE id to names, Drug Id to Names into files
    """
    # Save mapping from SE id to name into a file (SeIdNameC5.txt)
    id2ADr, dId2DrugName = utils.load_obj(params.ID2NamePath_TWOSIDEC5)
    fSE = open("%s/SeIdNameC5.txt" % DATASET_DIR, "w")
    for k in range(len(id2ADr)):
        v = id2ADr[k]
        fSE.write("%s\t%s\n" % (k, v))
    fSE.close()
    # Save mapping from drug id to name into a file (DrugId2NameC5.txt)
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
        # Set random seed
        resetRandomSeed()
        # Save Id-Name
        saveId2Name()
        writeSED()
        # Reset random seed
        resetRandomSeed()
        # Generating data
        exportFullTWOSIDEs()
    else:
        exportData()


if __name__ == "__main__":
    run()
