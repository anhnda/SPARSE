import matplotlib.pyplot as plt
import numpy as np
import params
import math
import torch
from dataFactory.dataLoader import DataLoader
from dataFactory.genData.genTWOSIDES import loadProtein2Pathway

from utils import utils

N_TOP = 400


def calPerson(matX, Y):
    r"""
    Calculate Pearson correlation for each column of matrixX with Y
    Formulation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Args:
        matX: shape of (nPoint, nDim)
        Y: shape of (nPoint)

    Returns:
        shape of (nDim)
    """
    Y = Y[:, np.newaxis]
    xy = matX * Y
    Exy = np.mean(xy, axis=0)
    Ex = np.mean(matX, axis=0)
    Ey = np.mean(Y, axis=0)
    dxy = Exy - Ex * Ey
    x2 = matX * matX
    Ex2 = np.mean(x2, axis=0)
    dx = np.sqrt(Ex2 - Ex * Ex)
    y2 = Y * Y
    Ey2 = np.mean(y2, axis=0)
    dy = np.sqrt(Ey2 - Ey * Ey)
    p = dxy / (dx * dy + 1e-10)
    return p


def loadProteinID2Name():
    r"""
    Load dictionary for protein integer id to protein name
    Returns:
    dictionary : {protein_integer_id: protein_uniport_name}
    """
    d = dict()
    try:
        fin = open("%s/DrugBank/UniProtein2Name.txt" % params.DATA_DIR)
        while True:
            line = fin.readline()
            if line == "":
                break
            parts = line.strip().split("|")
            d[parts[0]] = parts[1]
    except:
        print("No UniProtein2Name file")
        pass
    return d


def loadPathwayId2Name():
    r"""
    Load dictionary from pathway integer id to pathway text name

    Returns:
        num_pathway
        dictionary {pathway_integer_id: pathway_long_text_description_name}
        dictionary {pathway_kegg_name: pathway_integer_id}
        dictionary {proteinName: [list_of_pathway_kegg_names]
    """
    d = dict()
    dProtein2Pathway, dPathway2Id, dPathway2Name = loadProtein2Pathway()
    nPathway = len(dPathway2Id)
    dId2Pathway = utils.reverse_dict(dPathway2Id)
    for i in range(nPathway):
        pathway = dId2Pathway[i]
        pName = utils.get_dict(dPathway2Name, pathway, pathway)
        d[i] = pName
    return nPathway, d, utils.reverse_dict(dPathway2Id), dProtein2Pathway


def exportLatentFeature(dataPref="", pref="S", iFold=0, tau=params.Tau, shape=None):
    r"""
    Extracting top associated proteins/pathways features with each latent feature
    The output is saved to tmpOut/group2P.dat in the dictionary format and tmpOut/DrugLatentF*.txt in text format.

    (See Algorithm 1 for extracting latent features on the paper)
    """
    # Load protein integer id to protein name
    proteinId2Name = loadProteinID2Name()
    # Load latent drug features
    drugEmbeddings = np.loadtxt("%s/%s_%s_D_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
    # Load protein_uniport_id to integer_id, drug feature matrix
    dProtein2Id, drugFeatures = utils.load_obj("%s/TWOSIDES/TWOSIDESfeatures_1.dat" % params.TMP_DIR)
    # Save protein ids
    fpout = open("%s/ProteinIDList.txt" % params.TMP_DIR, "w")
    for p in list(sorted(dProtein2Id.keys())):
        fpout.write("%s\n" % p)
    fpout.close()
    # Create map from protein_integer_id to  protein_uniprot_id
    dId2Protein = utils.reverse_dict(dProtein2Id)
    # Load pathway name maps
    nPathWay, dPathwayId2Name, dPathwayId2KEGGId, _ = loadPathwayId2Name()
    nProtein = len(dProtein2Id)
    # Extract corresponding protein features and pathway feature section from drug feature matrix
    proteinFeatures = drugFeatures[:, -(nProtein + nPathWay): -nPathWay]
    pathWayFeatures = drugFeatures[:, -nPathWay:]

    print("Protein shape: ", proteinFeatures.shape, nProtein)
    print("Drug latent feature shape: ", drugEmbeddings.shape)
    nD, K = drugEmbeddings.shape
    assert proteinFeatures.shape[1] == nProtein
    assert proteinFeatures.shape[0] == nD
    # Path to save drug latent features
    pathLatentFeatures = "%s/DrugLatentF_%s_%s.txt" % (params.TMP_DIR, iFold, tau)
    fout = open(pathLatentFeatures, "w")

    # Threshold for number of top proteins and pathways for extraction (Algorithm 1)
    N_P = 20
    N_A = 10
    dGroup2PV = dict()
    print(nPathWay)
    # Iteration over K latent dimensions
    for k in range(K):
        # Drug latent features at dim k
        y = drugEmbeddings[:, k]
        # Calculating pearson correlations of protein features with drug latent features at dim k
        p = calPerson(proteinFeatures, y)
        # Extracting top proteins
        ps = np.argsort(p)[::-1][:N_P]
        ar = np.zeros(nProtein)
        ar[ps] = 1
        # Calculating pearson correlations of pathway features with drug latent features at dim k
        pa = calPerson(pathWayFeatures, y)
        # Extracting top pathways
        psa = np.argsort(pa)[::-1][:N_A]
        ara = np.zeros(nPathWay)
        ara[psa] = 1
        dGroup2PV[k] = (ar, ara)

        # Write corresponding extracting to files
        fout.write("Group_%s:\n" % k)
        fout.write("\tProteins: \n\n")
        for ii in ps:
            pid = dId2Protein[ii]
            fout.write("\t%s %s\n" % (pid, utils.get_dict(proteinId2Name, pid, pid)))
        fout.write("\tPathways: \n\n")
        for ii in psa:
            if pa[ii] > 0:
                pathway = dPathwayId2Name[ii]
                fout.write("\t%s %s\n" % (dPathwayId2KEGGId[ii], pathway))
    fout.close()
    utils.save_obj(dGroup2PV, "%s/group2P.dat" % params.TMP_DIR)
    if params.INFO_OUTPUT:
        print("Latent features extracted at: ", pathLatentFeatures)


def dOuterP(v1, v2, v3):
    r"""
    Outer product of 3 vectors: v1, v2, v3
    Args:
        v1: vector of dim K1
        v2: vector of dim K2
        v3: vector of dim K3

    Returns: A tensor for the outer product of dim K1 x K2 x K3

    """
    v12 = np.outer(v1, v2)
    v123 = np.outer(v12, v3)
    return v123.reshape((len(v1), len(v2), len(v3)))


def matOuterX(m1, m2, m3, b):
    r"""
    Tensor product of m1, m2, m3 and interaction matrix b
    Args:
        m1: input matrix 1
        m2: input matrix 2
        m3: input matrix 3
        b: interaction matrix

    Returns: Tensor dot product

    """
    v1 = np.tensordot(m1, b, axes=1)
    v2 = np.tensordot(m2, v1.transpose((1, 0, 2)), axes=1)
    v3 = np.tensordot(m3, v2.transpose((2, 0, 1)), axes=1)
    vx = v3.transpose((2, 1, 0))
    return vx


def swapMax(u, v):
    r"""
    Swap two number by an ascending order
    Args:
        u:
        v:

    Returns:

    """
    if u > v:
        u, v = v, u
    return u, v


def checkModel(iFold=0, tau=0.02, dataPref=""):
    r"""
    Check whether the trained model for iFold with horseshoe term tau of dataPref does exist or not
    Args:
        iFold: fold id
        tau: horseshoe tau
        dataPref: data prefix

    Returns: boolean value for the existence of the corresponding trained model

    """
    import os
    path = "%s/%s_%s_D_%s_%s_1.txt" % (params.TMP_DIR, dataPref, "S", iFold, tau)
    return os.path.exists(path)


def exportTopPredictionAll(dataPref="", pref="S", iFold=0, tau=params.Tau, shape=None, pName=True,
                           matchingPath="%s/PairMatching.txt" % params.TMP_DIR):
    r"""
    Exporting the top of all predictions
    See Algorithm 1 for detail
    """

    dPId2Name = loadProteinID2Name()
    # Load latent features of drugs, side effects, and latent interactions
    d = np.loadtxt(
        "%s/%s_%s_D_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
    e = np.loadtxt(
        "%s/%s_%s_S_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
    b = np.loadtxt(
        "%s/%s_%s_B_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))

    k = b.shape[0]

    suffix = "HS"

    if shape is not None:
        k1, k2, k3 = shape
    else:
        k1 = k2 = k3 = k
    # Reshape the latent interactions into a 3-dimensional tensor
    bTensor = b.reshape((k1, k2, k3))
    # Load mapping of names and ids for proteins
    dProtein2Id, features = utils.load_obj("%s/TWOSIDES/TWOSIDESfeatures_1.dat" % params.TMP_DIR)
    dId2Protein = utils.reverse_dict(dProtein2Id)
    # Load extracted proteins (pathways) associated with each latent feature
    dGroup2PV = utils.load_obj("%s/group2P.dat" % params.TMP_DIR)
    nProtein = len(dProtein2Id)
    # Load mapping of names and ids for pathways
    nPathway, dPathwayId2Name, dPathwayId2KEGGID, dProtein2Pathway = loadPathwayId2Name()

    # Load dataWrapper of fold i
    dataWrapper = DataLoader(forceCPU=True)
    dataWrapper.loadData(iFold)

    nD, nSe = dataWrapper.data.nD, dataWrapper.data.nSe
    # sId = np.random.randint(0, nSe) + 99
    id2ADr, dId2DrugName = utils.load_obj(params.ID2NamePath_TWOSIDEC5)

    def getPA(d1, k):
        r"""
        Get corresponding protein and pathways of drug d1 on dimension k
        The results are the matching of avaiable protein-pathways of drug1 with corresponding protein-pathways of dim k
        (Corresponding to lines with Non-zero features in Algorithm 1)
        Args:
            d1: drug integer id
            k: dimension id

        Returns:

        """
        # Get protein feature of drug d1
        xd = features[d1, -(nProtein + nPathway): -nPathway]
        # Get protein, pathway feature of dim k
        pk, pak = dGroup2PV[k]
        # Get pairwise dot product
        v = xd * pk
        # Get non-zero features
        pp = np.nonzero(v)[0]
        proteins = []
        paSet = set()
        #Extract names of non-zeros features of proteins
        for p in pp:
            e = dId2Protein[p]
            pathwayIds = utils.get_dict(dProtein2Pathway, e, [])
            for pathwayId in pathwayIds:
                paSet.add(pathwayId)

            if pName:
                e = utils.get_dict(dPId2Name, e, e)
            proteins.append(e)

        # Get pathways feature of drug d1
        xd2 = features[d1, -nPathway:]
        # Get pairwise product of drug feature with corresponding latent dim feature
        v2 = xd2 * pak
        # Get non-zero pathway features
        pp2 = np.nonzero(v2)[0]
        pathways = []
        # Extract corresonding pathway names
        for pa in pp2:
            if pa in paSet:
                e = dPathwayId2Name[pa]
                pathways.append(e)
        if len(pathways) == 0:
            pathways.append("Unknown")

        return proteins, pathways

    pathOut1 = "%s/RawInterpretation.txt" % (params.TMP_DIR)
    pathOut2 = "%s/TopPredictedTriples.txt" % (params.TMP_DIR)
    fout = open(pathOut1, "w")
    fout2 = open(pathOut2, "w")
    # Get all prediction
    pred = matOuterX(d, d, e, bTensor)
    # Exclude training samples
    mask = 1 - dataWrapper.ddiTensor
    print("DB")
    print(dataWrapper.ddiTensor.shape)
    print(pred.shape)
    print(d.shape, e.shape, bTensor.shape)
    # New prediction not in training data
    pred *= mask
    # Sorted by scores
    maxPredIndices = np.argsort(pred.reshape(-1))[::-1]

    nTopSamples = 10000
    nC = 0
    ress = []
    dSet = set()
    # Sample N_TOP predictions
    for i in range(nTopSamples):

        d1i, d2i, ei = np.unravel_index(maxPredIndices[i], (nD, nD, nSe))
        if d1i == d2i:
            continue
        sd1, sd2 = swapMax(d1i, d2i)
        # ONly add new interaction drug pair
        if (sd1, sd2) in dSet:
            continue
        dSet.add((sd1, sd2))
        nC += 1
        # Get the corresponding embedding of drug, side effects
        ed1, ed2, ee = d[d1i, :], d[d2i, :], e[ei, :]
        # Calculating interactions of latent dimensions
        sTripple = dOuterP(ed1, ed2, ee)
        sTripple *= bTensor
        # Get the highest interaction
        ranks = np.argsort(sTripple.reshape(-1))[::-1]
        # Append the highest latent interactions to the predicted triples
        i1, i2, i3 = np.unravel_index(ranks[0], shape=bTensor.shape)
        ress.append((d1i, d2i, ei, i1, i2, i3))
        if nC == N_TOP:
            break

    # dMatchingDrugPair = set()
    dDes = dict()
    fMatching = None
    fNoMatching = None
    # Save prediction triples
    if matchingPath is not None:
        fin = open(matchingPath)
        while True:
            line = fin.readline()
            if line == "":
                break
            parts = line.strip().split("||")
            dpair = parts[0]
            des = parts[1]
            dDes[dpair] = des
        fMatching = open("%s/InterPredMatching.txt" % params.TMP_DIR, "w")
        fNoMatching = open("%s/InterPredNoMatching.txt" % params.TMP_DIR, "w")

    def srt(d1, d2):
        if d1 > d2:
            d1, d2 = d2, d1
        return d1, d2

    ii1 = 0
    ii2 = 0
    # Extracting prediction with interpretation
    for res in ress:
        # Get triples (d1, d2, e) and corresponding latent dimension
        d1, d2, e3, i1, i2, i3 = res
        # Extracting protein-pathway of d1
        l1, la1 = getPA(d1, i1)
        # Extracting protein-pathways of d2
        l2, la2 = getPA(d2, i2)
        # Get drug, se names
        d1Name, d2Name, sName, l1, l2, la1, la2 = dId2DrugName[d1], dId2DrugName[d2], id2ADr[e3], l1, l2, la1, la2
        # Write prediction with interpretation
        if matchingPath is not None:
            d1x, d2x = srt(d1Name.lower(), d2Name.lower())
            dpair = "%s,%s" % (d1x, d2x)
            r = utils.get_dict(dDes, dpair, "")
            if r == "":
                ff = fNoMatching
                ii2 = ii2 + 1
                ii = ii2
            else:
                ff = fMatching
                ii1 = ii1 + 1
                ii = ii1
            ff.write("\n+%s) New sample: %s, %s, %s, %s, %s, %s\n" % (ii, d1Name, d2Name, sName, i1, i2, i3))
            if r != "":
                ff.write("\tDescription: %s\n" % r.replace(".", "\n"))
            ff.write("\t%s:\n \t-> Proteins: %s\n" % (d1Name, ",".join(l1)))
            ff.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la1))
            ff.write("\t%s:\n \t-> Proteins: %s\n" % (d2Name, ",".join(l2)))
            ff.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la2))

        else:
            fout.write("\n+) New sample: %s, %s, %s, %s, %s, %s\n" % (d1Name, d2Name, sName, i1, i2, i3))
            fout.write("\t%s:\n \t-> Proteins: %s\n" % (d1Name, ",".join(l1)))
            fout.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la1))
            fout.write("\t%s:\n \t-> Proteins: %s\n" % (d2Name, ",".join(l2)))
            fout.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la2))
            fout2.write("%s, %s, %s\n" % (d1Name, d2Name, sName))

    fout.close()
    if matchingPath is not None:
        fMatching.close()
        fNoMatching.close()

    if params.INFO_OUTPUT:
        print("Top predictions are at: ", pathOut2)
        print("Raw interpretations are at: ", pathOut1)


def exportTopPredictionEachSE(dataPref="", pref="S", iFold=0, tau=params.Tau, pName=True,
                              matchingPath="%s/PairMatching.txt" % params.TMP_DIR, shape=None):
    r"""
    Exporting top prediction for each side effect.
    It is similar to all predictions, except that the ranking is grouped for each side effect
    """

    dPId2Name = loadProteinID2Name()
    d = np.loadtxt(
        "%s/%s_%s_D_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
    e = np.loadtxt(
        "%s/%s_%s_S_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
    b = np.loadtxt(
        "%s/%s_%s_B_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))

    k = b.shape[0]

    if shape is not None:
        k1, k2, k3 = shape
    else:
        k1 = k2 = k3 = k

    bTensor = b.reshape((k1, k2, k3))

    dProtein2Id, features = utils.load_obj("%s/TWOSIDES/TWOSIDESfeatures_1.dat" % params.TMP_DIR)
    dId2Protein = utils.reverse_dict(dProtein2Id)
    dGroup2PV = utils.load_obj("%s/group2P.dat" % params.TMP_DIR)
    nProtein = len(dProtein2Id)

    nPathway, dPathwayId2Name, dPathwayId2KEGGID, dProtein2Pathway = loadPathwayId2Name()

    dataWrapper = DataLoader(forceCPU=True)
    dataWrapper.loadData(iFold)

    nD, nSe = dataWrapper.data.nD, dataWrapper.data.nSe
    # sId = np.random.randint(0, nSe) + 99
    id2ADr, dId2DrugName = utils.load_obj(params.ID2NamePath_TWOSIDEC5)

    def getProteinPathways(drugId, latentFeatureId):
        xd = features[drugId, -(nProtein + nPathway): -nPathway]
        pk, pak = dGroup2PV[latentFeatureId]
        v = xd * pk
        pp = np.nonzero(v)[0]
        proteins = []
        pathwaySet = set()
        for p in pp:
            e = dId2Protein[p]
            pathwayIds = utils.get_dict(dProtein2Pathway, e, [])
            for pathwayId in pathwayIds:
                pathwaySet.add(pathwayId)

            if pName:
                e = utils.get_dict(dPId2Name, e, e)
            proteins.append(e)

        xd2 = features[drugId, -nPathway:]
        v2 = xd2 * pak
        pp2 = np.nonzero(v2)[0]
        pathways = []
        for pa in pp2:
            if pa in pathwaySet:
                e = dPathwayId2Name[pa]
                pathways.append(e)
        if len(pathways) == 0:
            pathways.append("Unknown")

        return proteins, pathways

    pathOut1 = "%s/RawInterpretation.txt" % (params.TMP_DIR)
    pathOut2 = "%s/TopPredictedTriples.txt" % (params.TMP_DIR)
    fout = open(pathOut1, "w")
    fout2 = open(pathOut2, "w")

    predScores = matOuterX(d, d, e, bTensor)

    # Remove known interactions by using a mask
    mask = 1 - dataWrapper.ddiTensor
    predScores *= mask

    # Extracting top predictions by SEs
    topTriples = []
    for ii in range(nSe):
        maxIdxs = np.argsort(predScores[:, :, ii].reshape(-1))[::-1]
        for jj in range(10):
            d1, d2 = np.unravel_index(maxIdxs[jj], (nD, nD))
            if d1 == d2:
                continue
            topTriples.append((d1, d2, ii))

    nC = 0
    ress = []
    dSet = set()
    for tp in topTriples:

        d1i, d2i, ei = tp
        sd1, sd2 = swapMax(d1i, d2i)
        if (sd1, sd2) in dSet:
            continue
        dSet.add((sd1, sd2))
        nC += 1

        ed1, ed2, ee = d[d1i, :], d[d2i, :], e[ei, :]
        sTripple = dOuterP(ed1, ed2, ee)
        sTripple *= bTensor
        ranks = np.argsort(sTripple.reshape(-1))[::-1]
        i1, i2, i3 = np.unravel_index(ranks[0], shape=bTensor.shape)
        ress.append((d1i, d2i, ei, i1, i2, i3))
        if nC == N_TOP:
            break
    dDes = dict()
    fMatching = None
    fNoMatching = None
    if matchingPath is not None:
        fin = open(matchingPath)
        while True:
            line = fin.readline()
            if line == "":
                break
            parts = line.strip().split("||")
            dpair = parts[0]
            des = parts[1]
            dDes[dpair] = des
        fMatching = open("%s/InterPredMatching.txt" % params.TMP_DIR, "w")
        fNoMatching = open("%s/InterPredNoMatching.txt" % params.TMP_DIR, "w")

    def srt(d1, d2):
        if d1 > d2:
            d1, d2 = d2, d1
        return d1, d2

    ii1 = 0
    ii2 = 0

    for res in ress:
        d1, d2, e3, i1, i2, i3 = res
        l1, la1 = getProteinPathways(d1, i1)
        l2, la2 = getProteinPathways(d2, i2)
        d1Name, d2Name, sName, l1, l2, la1, la2 = dId2DrugName[d1], dId2DrugName[d2], id2ADr[e3], l1, l2, la1, la2

        if matchingPath is not None:
            d1x, d2x = srt(d1Name.lower(), d2Name.lower())
            dpair = "%s,%s" % (d1x, d2x)
            r = utils.get_dict(dDes, dpair, "")
            if r == "":
                ff = fNoMatching
                ii2 = ii2 + 1
                ii = ii2
            else:
                ff = fMatching
                ii1 = ii1 + 1
                ii = ii1
            ff.write("\n+%s) New prediction: %s, %s, %s, %s, %s, %s\n" % (ii, d1Name, d2Name, sName, i1, i2, i3))
            if r != "":
                ff.write("\tDescription: %s\n" % r.replace(".", "\n"))
            ff.write("\t%s:\n \t-> Proteins: %s\n" % (d1Name, ",".join(l1)))
            ff.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la1))
            ff.write("\t%s:\n \t-> Proteins: %s\n" % (d2Name, ",".join(l2)))
            ff.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la2))

        else:
            fout.write("\n+) New sample: %s, %s, %s, %s, %s, %s\n" % (d1Name, d2Name, sName, i1, i2, i3))
            fout.write("\t%s:\n \t-> Proteins: %s\n" % (d1Name, ",".join(l1)))
            fout.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la1))
            fout.write("\t%s:\n \t-> Proteins: %s\n" % (d2Name, ",".join(l2)))
            fout.write("\t-> Pathways: \n\t\t%s\n" % "\n\t \t".join(la2))
            fout2.write("%s, %s, %s\n" % (d1Name, d2Name, sName))

    fout.close()
    if matchingPath is not None:
        fMatching.close()
        fNoMatching.close()
    if params.INFO_OUTPUT:
        print("Top predictions are at: ", pathOut2)
        print("Raw interpretations are at: ", pathOut1)

#
# def matchTopListDrugsCom(dataPref="", pref="S", iFold=0, tau=params.Tau, shape=None,
#                          pName=True,
#                          pairMatchingPath="%s/PairMatching.txt" % params.TMP_DIR,
#                          explanationPath="%s/Top10Explanation.txt" % params.TMP_DIR):
#
#     r"""
#     Merging the top prediction with the response from drugs.com interaction checker
#     Args:
#         dataPref: Data prefix of the dataset (Default: "")
#         pref: Prefix for the output file
#         iFold: FOld id
#         tau: horseshoe term tau
#         shape: shape of latent dims
#         pName: use protein names
#         pairMatchingPath: matching pairs of interactions
#         explanationPath: expert manual explanation path
#
#     Returns:
#
#     """
#     dPId2Name = loadProteinID2Name()
#     drugEmbeddings = np.loadtxt(
#         "%s/%s_%s_D_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
#     seEmbeddings = np.loadtxt(
#         "%s/%s_%s_S_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
#     latentInteractions = np.loadtxt(
#         "%s/%s_%s_B_%s_%s_1.txt" % (params.TMP_DIR, dataPref, pref, iFold, tau))
#
#     latentSize = latentInteractions.shape[0]
#     suffix = "HS"
#
#     if shape is not None:
#         k1, k2, k3 = shape
#     else:
#         k1 = k2 = k3 = latentSize
#
#     bTensor = latentInteractions.reshape((k1, k2, k3))
#
#     dProtein2Id, features = utils.load_obj("%s/TWOSIDES/TWOSIDESfeatures_1.dat" % params.TMP_DIR)
#     dId2Protein = utils.reverse_dict(dProtein2Id)
#     dGroup2PV = utils.load_obj("%s/group2P.dat" % params.TMP_DIR)
#     nProtein = len(dProtein2Id)
#
#     nPathway, dPathwayId2Name, dPathwayId2KEGGID, dProtein2Pathway = loadPathwayId2Name()
#
#     dataWrapper = DataLoader(forceCPU=True)
#     dataWrapper.loadData(iFold)
#
#     nD, nSe = dataWrapper.data.nD, dataWrapper.data.nSe
#     id2ADr, dId2DrugName = utils.load_obj(params.ID2NamePath_TWOSIDEC5)
#
#     def getPA(d1, k):
#         xd = features[d1, -(nProtein + nPathway): -nPathway]
#         pk, pak = dGroup2PV[k]
#         v = xd * pk
#         pp = np.nonzero(v)[0]
#         proteins = []
#         paSet = set()
#         for p in pp:
#             e = dId2Protein[p]
#             pathwayIds = utils.get_dict(dProtein2Pathway, e, [])
#             for pathwayId in pathwayIds:
#                 paSet.add(pathwayId)
#
#             if pName:
#                 e = utils.get_dict(dPId2Name, e, e)
#             proteins.append(e)
#
#         xd2 = features[d1, -nPathway:]
#         v2 = xd2 * pak
#         pp2 = np.nonzero(v2)[0]
#         pathways = []
#         for pa in pp2:
#             if pa in paSet:
#                 e = dPathwayId2Name[pa]
#                 pathways.append(e)
#         if len(pathways) == 0:
#             pathways.append("Unknown")
#
#         return proteins, pathways
#
#     fout = open("%s/InterPredictionX_%s.txt" % (params.TMP_DIR, suffix), "w")
#
#     pred = matOuterX(drugEmbeddings, drugEmbeddings, seEmbeddings, bTensor)
#     mask = 1 - dataWrapper.ddiTensor
#     print("DB")
#     print(dataWrapper.ddiTensor.shape)
#     print(pred.shape)
#     print(drugEmbeddings.shape, seEmbeddings.shape, bTensor.shape)
#     pred *= mask
#
#     topTriples = []
#     for ii in range(nSe):
#         maxIdxs = np.argsort(pred[:, :, ii].reshape(-1))[::-1]
#         for jj in range(10):
#             d1, d2 = np.unravel_index(maxIdxs[jj], (nD, nD))
#             if d1 == d2:
#                 continue
#
#             topTriples.append((d1, d2, ii))
#
#     nC = 0
#     ress = []
#     dSet = set()
#     for tp in topTriples:
#
#         d1i, d2i, ei = tp
#         sd1, sd2 = swapMax(d1i, d2i)
#         if (sd1, sd2) in dSet:
#             continue
#         dSet.add((sd1, sd2))
#         nC += 1
#
#         ed1, ed2, ee = drugEmbeddings[d1i, :], drugEmbeddings[d2i, :], seEmbeddings[ei, :]
#         sTripple = dOuterP(ed1, ed2, ee)
#         sTripple *= bTensor
#         ranks = np.argsort(sTripple.reshape(-1))[::-1]
#         i1, i2, i3 = np.unravel_index(ranks[0], shape=bTensor.shape)
#         ress.append((d1i, d2i, ei, i1, i2, i3))
#         if nC == N_TOP:
#             break
#     print("N Top List: ", nC, ress[-1])
#     dDes = dict()
#     fMatching = None
#     fNoMatching = None
#     explanations = []
#     spareMatchingPath = "%s/SPARE_TopPredictions.txt" % params.TMP_DIR
#     if explanationPath is not None:
#         fExplanation = open(explanationPath)
#         lines = fExplanation.readlines()
#         for line in lines:
#             explanations.append(line.strip())
#     if pairMatchingPath is not None:
#         fin = open(pairMatchingPath)
#         while True:
#             line = fin.readline()
#             if line == "":
#                 break
#             parts = line.strip().split("||")
#             dpair = parts[0]
#             des = parts[1]
#             dDes[dpair] = des
#         fMatching = open(spareMatchingPath, "w")
#         fNoMatching = open("%s/InterNoMatching.tsv" % params.TMP_DIR, "w")
#
#     def srt(d1, d2):
#         if d1 > d2:
#             d1, d2 = d2, d1
#         return d1, d2
#
#     ii1 = 0
#     ii2 = 0
#
#     for res in ress:
#         d1, d2, e3, i1, i2, i3 = res
#         l1, la1 = getPA(d1, i1)
#         l2, la2 = getPA(d2, i2)
#         d1Name, d2Name, sName, l1, l2, la1, la2 = dId2DrugName[d1], dId2DrugName[d2], id2ADr[e3], l1, l2, la1, la2
#
#         if pairMatchingPath is not None:
#             d1x, d2x = srt(d1Name.lower(), d2Name.lower())
#             dpair = "%s,%s" % (d1x, d2x)
#             r = utils.get_dict(dDes, dpair, "")
#             if r == "":
#                 ff = fNoMatching
#                 ii2 = ii2 + 1
#                 ii = ii2
#             else:
#                 ff = fMatching
#                 ii1 = ii1 + 1
#                 ii = ii1
#             ff.write("+%s) Predicted interaction: %s, %s, %s\n" % (ii, d1Name, d2Name, sName))
#             # ff.write("%s\t%s\t%s\t%s" % (ii, d1Name, d2Name, sName))
#             if r != "":
#                 ff.write("\tDescription: %s\n" % r.replace(".", "\n"))
#             ff.write("- For drug %s: \n\t\tProteins: %s \n\t\tPathways: %s\n" % (d1Name, ",".join(l1), ",".join(la1)))
#             ff.write("- For drug %s: \n\t\tProteins: %s \n\t\tPathways: %s\n" % (d2Name, ",".join(l2), ",".join(la2)))
#
#             if ii <= 10 and r != "" and explanationPath is not None:
#                 ff.write("- Explanation: %s\n" % explanations[ii - 1])
#             else:
#                 ff.write("\t\n")
#             ff.write("\n")
#         else:
#             print("Need TripleMatching file.")
#     fout.close()
#     if pairMatchingPath is not None:
#         fMatching.close()
#         fNoMatching.close()
#
#     if params.INFO_OUTPUT:
#         if pairMatchingPath is not None:
#             print("Top matching predictions: ", spareMatchingPath)
#

def matchTopListDrugsComX(dataPref="", pref="S", iFold=0, tau=params.Tau, shape=None,
                         pName=True,
                         rawInterpretationPath = "%s/RawInterpretation.txt"  % params.TMP_DIR,
                         pairMatchingPath="%s/PairMatching.txt" % params.TMP_DIR,
                         explanationPath="%s/Top10Explanation.txt" % params.TMP_DIR):



    """
    Args:
        dataPref: Data prefix of the dataset (Default: "")
        pref: Prefix for the output file
        iFold: FOld id
        tau: horseshoe term tau
        shape: shape of latent dims
        pName: use protein names
        rawInterpretationPath: Raw interpretation extracted from all predictions or for each side effects
        pairMatchingPath: matching pairs of interactions
        explanationPath: expert manual explanation path

    Returns:

    """
    dDes = dict()
    explanations = []
    # Path to final matching file with drugs.com descriptions
    spareMatchingPath = "%s/SPARE_TopPredictions.txt" % params.TMP_DIR
    if explanationPath is not None:
        fExplanation = open(explanationPath)
        lines = fExplanation.readlines()
        for line in lines:
            explanations.append(line.strip())
    assert pairMatchingPath is not None

    # Load drugs.com descriptions for interacting drug pairs
    fin = open(pairMatchingPath)
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("||")
        dpair = parts[0]
        des = parts[1]
        dDes[dpair] = des
    fMatching = open(spareMatchingPath, "w")
    fNoMatching = open("%s/InterNoMatching.tsv" % params.TMP_DIR, "w")

    def srt(d1, d2):
        if d1 > d2:
            d1, d2 = d2, d1
        return d1, d2
    # Load raw intepretation from RawInterpretation.txt
    def loadRawPrediction():
        fin = open(rawInterpretationPath)
        contentLIst = []
        tripleList = []
        #Skip first line
        fin.readline()
        currentContent = []
        while True:
            line = fin.readline()
            if line == "":
                break
            if line.startswith("+)"):
                if len (currentContent) > 0:
                    contentLIst.append(currentContent)
                    currentContent = []
                parts = line.strip().split(":")[1]
                drugs = parts.split(",")
                d1 = drugs[0].strip()
                d2 = drugs[1].strip()
                se = drugs[2].strip()
                tripleList.append([d1, d2, se])
                currentContent.append("+)Prediction: %s,%s,%s\n" % (d1,d2,se))
            else:
                currentContent.append(line)
        if len(currentContent) > 0:
            contentLIst.append(currentContent)
        return contentLIst, tripleList

    contentList, tripleList = loadRawPrediction()
    ii1 = 0
    ii2 = 0
    # Merging raw prediction interpretation with description from drugs.com

    for i, tp in enumerate(tripleList):
        d1Name, d2Name, _ = tp
        content = contentList[i]
        d1x, d2x = srt(d1Name.lower(), d2Name.lower())
        dpair = "%s,%s" % (d1x, d2x)
        r = utils.get_dict(dDes, dpair, "")
        if r == "":
            ff = fNoMatching
            ii2 = ii2 + 1
            ii = ii2
        else:
            ff = fMatching
            ii1 = ii1 + 1
            ii = ii1
        ff.write("%s" % content[0])
        # ff.write("%s\t%s\t%s\t%s" % (ii, d1Name, d2Name, sName))

        if r != "":
            ff.write("\tDescription: %s\n" % r.replace(".", "\n"))
        ff.write("%s"%"".join(content[1:]))

        if ii <= 10 and r != "" and explanationPath is not None:
            ff.write("- Explanation: %s\n" % explanations[ii - 1])
        else:
            ff.write("\t\n")
        ff.write("\n")


    fMatching.close()
    fNoMatching.close()

    if params.INFO_OUTPUT:
        if pairMatchingPath is not None:
            print("Top matching predictions: ", spareMatchingPath)
def extract(tau=0.02, mode=1):
    # Export associated protein-pathways with each latent features
    exportLatentFeature(tau=tau)
    if mode == 1:
        # Export top prediction for each side effect
        exportTopPredictionEachSE(tau=tau, matchingPath=None)
    else:
        # Export top prediction for all side effects
        exportTopPredictionAll(tau=tau, matchingPath=None)


def rematching(tau=0.02):
    # Rematch interpretation of top predictions from SPARSE with drugs.com interaction checkers
    matchTopListDrugsComX(tau=tau, explanationPath=None)


if __name__ == "__main__":
    tau = 0.02
    params.HIGH_TWOSIDES = True
    extract()
    rematching(tau)
