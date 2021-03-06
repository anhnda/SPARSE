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
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pass

    def loadData(self, iFold, dataPref=""):
        self.iFold = iFold
        folder = "%s/TWOSIDES" % params.TMP_DIR
        if dataPref == "C":
            folder = "%s/CADDDI" % params.TMP_DIR
        elif dataPref == "J":
            folder = "%s/JADERDDI" % params.TMP_DIR

        print("Loading iFold: ", iFold)
        # print("Folder: ", folder)
        all_p = ""
        if dataPref == "":
            if params.HIGH_TWOSIDES:
                assert iFold == 0
                all_p = "fullC5"
        dataPath = "%s/%s%s_%d_%d_%d_%s" % (
            folder, all_p, params.D_PREF, params.MAX_R_ADR, params.MAX_R_DRUG, params.ADR_OFFSET, iFold)
        print("Data path: ", dataPath)
        data = utils.load_obj(dataPath)
        print(data.nD, data.nSe)
        ddiTensor = np.zeros((data.nD, data.nD, data.nSe))

        train2Label = data.pTrainPair2Label
        test2Label = data.pTestPosLabel
        valid2Label = data.pValidPosLabel
        negTest2Label = data.pTestNegLabel

        indices = []
        print("Cont.")
        for edge, label in train2Label.items():
            d1, d2 = edge
            for lb in label:
                indices.append((d1, d2, lb))
                indices.append((d2, d1, lb))
        ddiTensor[tuple(np.transpose(indices))] = 1

        testPosIndices = []
        validPosIndices = []
        testNegIndices = []

        for edge, label in test2Label.items():
            d1, d2 = edge
            for lb in label:
                testPosIndices.append((d1, d2, lb))
                # testPosIndices.append((d2, d1, l))

        testPosIndices = tuple(np.transpose(testPosIndices))

        for edge, label in valid2Label.items():
            d1, d2 = edge
            for lb in label:
                validPosIndices.append((d1, d2, lb))
                # validPosIndices.append((d2, d1, l))

        validPosIndices = tuple(np.transpose(validPosIndices))
        for edge, label in negTest2Label.items():
            d1, d2 = edge
            for lb in label:
                testNegIndices.append((d1, d2, lb))

        testNegIndices = tuple(np.transpose(testNegIndices))

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
            self.hyperEdgeIndex = data.hyperEdgeCliqueIndex.to(self.device)
            self.hyperEdgeTypes = torch.from_numpy(data.hyperEdgeIndexType).long().to(self.device)


class RealData:
    def __init__(self, trainFolds, testFolds, validFolds, AFolds, UAFolds, negFold, features):
        self.trainFolds = trainFolds
        self.testFolds = testFolds
        self.validFolds = validFolds
        self.AFolds = AFolds
        self.UAFolds = UAFolds
        self.negFold = negFold
        self.drug2Features = features

        self.featureSize = self.drug2Features.shape[1]


class RealFoldData:
    def __init__(self, trainFold, testFold, validFold, AFold, UAFold, negFold, features):
        self.trainFold = trainFold
        self.testFold = testFold
        self.validFold = validFold
        self.AFold = AFold
        self.UAFold = UAFold
        self.negFold = negFold
        self.drug2Features = features
        self.featureSize = self.drug2Features.shape[1]


def loadDrugProteinMap(skipBDB=True):
    from utils.utils import loadMapSetFromFile

    def mergeDict(d1, d2):
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

    dDrugProteinDrugBank = loadMapSetFromFile(params.DRUGBANK_INCHI_PROTEIN, "|", sepValue=",")
    if skipBDB:
        dDrugProteinBindingDB = {}
    else:
        # Load BindingDB:
        dDrugProteinBindingDB = loadMapSetFromFile(params.BINDINGDB_EXTENDED)
    dd = mergeDict(dDrugProteinBindingDB, dDrugProteinDrugBank)

    return dd


def loadProtein2Pathway():
    dProtein2Pathway = loadMapSetFromFile(params.KEGG_UNIPROT_PATHWAY_MAPPING_PATH, sep="|", sepValue=",")
    return dProtein2Pathway
