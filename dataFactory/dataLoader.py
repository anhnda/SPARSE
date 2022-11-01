import numpy as np
from utils import utils
import params
import torch

from utils.utils import loadMapSetFromFile


class DataLoader:
    """
    DataLoader loads each fold data given the dataset and the fold Id
    """
    def __init__(self, forceCPU=False):

        if forceCPU:
            # Force to run on CPU
            self.device = torch.device('cpu')
        else:
            # Check if cuda is avalable then use cuda, otherwise, use cpu
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pass

    def loadData(self, iFold, dataPref=""):
        r"""
        Load data at fold i
        Args:
            iFold: fold i
            dataPref: prefix of data set: "" for TWOSIDES, "J" for JADERDDI, "C" for CADDDI

        """
        self.iFold = iFold
        # Check the corresponding dataset directory from Data Prefix
        # TWOSIDES by default
        folder = "%s/TWOSIDES" % params.TMP_DIR

        if dataPref == "C":
            # CADDDI
            folder = "%s/CADDDI" % params.TMP_DIR
        elif dataPref == "J":
            # JADERDDI
            folder = "%s/JADERDDI" % params.TMP_DIR

        print("Loading iFold: ", iFold)
        # print("Folder: ", folder)
        all_p = ""
        if dataPref == "":
            if params.HIGH_TWOSIDES:
                assert iFold == 0
                all_p = "fullC5"

        # RealFoldData dump path of the corresponding dataset
        dataPath = "%s/%s%s_%d_%d_%d_%s" % (
            folder, all_p, params.D_PREF, params.MAX_R_ADR, params.MAX_R_DRUG, params.ADR_OFFSET, iFold)
        print("Data path: ", dataPath)
        # Load RealFoldData instance (See dataFactor.dataLoader.RealFoldData)
        data = utils.load_obj(dataPath)
        print(data.nD, data.nSe)
        # Init DDI tensor of size |V_D| x |V_D| x |V_S|
        ddiTensor = np.zeros((data.nD, data.nD, data.nSe))

        # Positive Labels for training drug-pairs
        train2Label = data.pTrainPair2Label
        # Positive Labels for testing drug-pairs
        test2Label = data.pTestPosLabel
        # Positive Labels for validating drug-pairs
        valid2Label = data.pValidPosLabel
        # Hold-out labels for negative sampled drug pairs
        negTest2Label = data.pTestNegLabel

        indices = []
        print("Cont.")
        # Assign value for DDI tensor from positive training labels
        for edge, label in train2Label.items():
            d1, d2 = edge
            for lb in label:
                indices.append((d1, d2, lb))
                indices.append((d2, d1, lb))
        ddiTensor[tuple(np.transpose(indices))] = 1

        # List of triples of indices of positive drug-drug-side effect (hypereges)  for test and validation
        testPosIndices = []
        validPosIndices = []
        # List of triples of indices of holdout negative drug-drug-side effect (hypereges)

        testNegIndices = []
        # Get triples of drug-drug-side effect for test set
        for edge, label in test2Label.items():
            d1, d2 = edge
            for lb in label:
                testPosIndices.append((d1, d2, lb))
                # testPosIndices.append((d2, d1, l))

        testPosIndices = tuple(np.transpose(testPosIndices))
        # Get triples of drug-drug-side effect for validation
        for edge, label in valid2Label.items():
            d1, d2 = edge
            for lb in label:
                validPosIndices.append((d1, d2, lb))
                # validPosIndices.append((d2, d1, l))

        validPosIndices = tuple(np.transpose(validPosIndices))


        # Get triples of drug-drug-side effect for testing

        for edge, label in negTest2Label.items():
            d1, d2 = edge
            for lb in label:
                testNegIndices.append((d1, d2, lb))

        testNegIndices = tuple(np.transpose(testNegIndices))

        # Assign data for datawrapper with DDI tensors, drug features, triples of indices of test-valid-holdout
        self.ddiTensor = ddiTensor
        features = data.drug2Features
        if not params.PROTEIN_FEATURE:
            features = data.drug2Features[:, :881]
        self.features = torch.from_numpy(features).float().to(self.device)
        self.ddiTensorInDevice = torch.from_numpy(ddiTensor).float().to(self.device)
        self.testNegIndices = testNegIndices
        self.validPosIndices = validPosIndices
        self.testPosIndices = testPosIndices
        self.data = data

        if not params.FAST_TRAINING:
            # Use clique expansion data
            self.hyperEdgeIndex = data.hyperEdgeCliqueIndex.to(self.device)
            self.hyperEdgeTypes = torch.from_numpy(data.hyperEdgeIndexType).long().to(self.device)

#
# class RealData:
#     def __init__(self, trainFolds, testFolds, validFolds, AFolds, UAFolds, negFold, features):
#         r"""
#
#         Args:
#             trainFolds:
#             testFolds:
#             validFolds:
#             AFolds:
#             UAFolds:
#             negFold:
#             features:
#         """
#         self.trainFolds = trainFolds
#         self.testFolds = testFolds
#         self.validFolds = validFolds
#         self.AFolds = AFolds
#         self.UAFolds = UAFolds
#         self.negFold = negFold
#         self.drug2Features = features
#
#         self.featureSize = self.drug2Features.shape[1]


class RealFoldData:
    def __init__(self, trainFold, testFold, validFold, AFold, UAFold, negFold, features):
        r"""
        Wrapper for fold data
        Args:
            trainFold: List of triples of indices of drug-drug-se for training
            testFold: List of triples of indices of drug-drug-se for testing
            validFold:  List of triples of indices of drug-drug-se for validating
            AFold: Adjacency matrix for the fold (For Baseline methods)
            UAFold: (Normalize Adjacency matrix of the fold (for baseline methods)
            negFold: List of triples of holdout negative of drug-drug-se
            features: Drug feature matrix
        """
        self.trainFold = trainFold
        self.testFold = testFold
        self.validFold = validFold
        self.AFold = AFold
        self.UAFold = UAFold
        self.negFold = negFold
        self.drug2Features = features
        self.featureSize = self.drug2Features.shape[1]


def loadDrugProteinMap(skipBDB=True):
    r"""
    Load drug-protein
    Args:
        skipBDB: Flag for using BindingDB dataset

    Returns:
    dictionary: {drugInchiKey: [List_of_ProteinUniProts]}

    """
    # Import a function to load mapping from key to list of values from a file
    from utils.utils import loadMapSetFromFile

    def mergeDict(d1, d2):
        r"""
        This function is used to merge two dictionaries
        Args:
            d1: dictionary 1
            d2: dictionary 2

        Returns: merged dictionary

        """
        from utils import utils
        d = {}
        for k1, v1 in d1.items():
            vm = set()
            v2 = utils.get_dict(d2, k1, set())
            for vi in (v1, v2):
                for vs in vi:
                    if vs != "":
                        vm.add(vs)
            d[k1] = vm
        for k2, v2 in d2.items():
            if k2 not in d1:
                vm = set()
                for vs in v2:
                    if vs != "":
                        vm.add(vs)
                d[k2] = vm
        return d

    # Load protein associated with each drug inchikey  from DrugBank
    dDrugProteinDrugBank = loadMapSetFromFile(params.DRUGBANK_INCHI_PROTEIN, "|", sepValue=",")
    if skipBDB:
        dDrugProteinBindingDB = {}
    else:
        # Load protein associated with each drug inchikey  from BindingDB
        dDrugProteinBindingDB = loadMapSetFromFile(params.BINDINGDB_EXTENDED)
    # Combine two databases
    dd = mergeDict(dDrugProteinBindingDB, dDrugProteinDrugBank)

    return dd


def loadProtein2Pathway():
    r"""
    Load pathways associated with each protein from KEGG database.

    Returns:
        dictionary {protein_uniport_id: [List_of_kegg_pathways]}

    """
    dProtein2Pathway = loadMapSetFromFile(params.KEGG_UNIPROT_PATHWAY_MAPPING_PATH, sep="|", sepValue=",")
    return dProtein2Pathway
