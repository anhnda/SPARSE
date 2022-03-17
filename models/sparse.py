import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import params
from scipy.stats import halfcauchy

from models.hyperconv import HyperConv


class SPARSECore(torch.nn.Module):
    r"""
    Sparse core module with Encoder - Decoder
    """
    def __init__(self, numDrug, numSE, latentSize, features=None, n_layer=1, device=torch.device('cpu'),
                 latentSizeList=None):
        super(SPARSECore, self).__init__()
        self.numDrug = numDrug
        self.numSE = numSE
        self.latentSize = latentSize
        self.device = device
        self.n_layer = n_layer
        latentSize1 = latentSize
        latentSize2 = latentSize
        if latentSizeList is not None and len(latentSizeList) > 0:
            latentSize1, latentSize2 = latentSizeList
            self.latentSize = latentSize1

        # Two layers for transforming drug features
        self.neuralList = torch.nn.ModuleList()
        self.neuralList.append(torch.nn.Linear(features.shape[-1], 2 * latentSize1).to(device))
        self.neuralList.append(torch.nn.Linear(2 * latentSize1, latentSize1).to(device))
        self.neuralAct = torch.nn.ReLU()

        self.convs = torch.nn.ModuleList()
        self.convAct = torch.nn.ReLU()
        for i in range(self.n_layer):
            layer = HyperConv(latentSize, latentSize, 5, device=device)
            self.convs.append(layer)

        self.seEmbeddings = torch.nn.Embedding(numSE, latentSize2).to(device)
        self.seIndices = torch.arange(0, numSE, dtype=torch.long).to(device)

        self.latentInteractions = torch.nn.Parameter(
            torch.rand((latentSize1, latentSize1, latentSize2), requires_grad=True).to(device))

        torch.nn.init.xavier_uniform_(self.seEmbeddings.weight)
        torch.nn.init.xavier_uniform_(self.latentInteractions.data)

        self.drugFeatures = features

        lambdaHorseShoe = halfcauchy.rvs(size=latentSize1 * latentSize1 * latentSize2)
        lambdaHorseShoe = lambdaHorseShoe.reshape((latentSize1, latentSize1, latentSize2))
        self.lambdaHorseShoe = torch.nn.Parameter(torch.from_numpy(lambdaHorseShoe).float(), requires_grad=True).to(
            device)

    def project(self):
        self.latentInteractions.data[self.latentInteractions.data < 0] = 0
        self.lambdaHorseShoe.data[self.lambdaHorseShoe.data < 0] = 0
        self.seEmbeddings.weight.data[self.seEmbeddings.weight.data < 0] = 0

    def encode1(self, drug1Indices, drug2Indices, seIndices):
        drugLatentFeatures = self.drugFeatures
        for nnLayer in self.neuralList:
            drugLatentFeatures = self.neuralAct(nnLayer(drugLatentFeatures))
        drug1LatentFeatures = drugLatentFeatures[drug1Indices]
        drug2LatentFeatures = drugLatentFeatures[drug2Indices]
        seLatentFeatures = self.seEmbeddings(seIndices)
        return drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures, drugLatentFeatures

    def decode1(self, drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures):
        v = torch.tensordot(drug1LatentFeatures, self.latentInteractions, dims=1)
        v = torch.tensordot(drug2LatentFeatures, v.transpose(1, 0), dims=1)
        v = torch.tensordot(seLatentFeatures, v.transpose(2, 0), dims=1)
        v = v.transpose(0, 1).transpose(1, 2)
        return v

    def encode2(self, edgeIndex, edgeTypes):
        drugLatentFeatures = self.drugFeatures
        for i, layer in enumerate(self.neuralList):
            drugLatentFeatures = layer(drugLatentFeatures)
            if i < len(self.neuralList) - 1:
                drugLatentFeatures = self.neuralAct(drugLatentFeatures)
        seLatentFeatures = self.seEmbeddings(self.seIndices)
        allLatentFeatures = torch.vstack((drugLatentFeatures, seLatentFeatures))
        for layer in self.convs:
            allLatentFeatures = self.convAct(layer(allLatentFeatures, edgeIndex, edgeTypes))
        return allLatentFeatures

    def decode2(self, allLatentFeatures, drug1Indices, drug2Indices, seIndices):

        drugEmbeddings = allLatentFeatures[:self.numDrug, ]
        seEmbeddings = allLatentFeatures[self.numDrug:, ]
        drug1LatentFeatures = drugEmbeddings[drug1Indices]
        drug2LatentFeatures = drugEmbeddings[drug2Indices]
        seLatentFeatures = seEmbeddings[seIndices]

        v = torch.tensordot(drug1LatentFeatures, self.latentInteractions, dims=1)
        v = torch.tensordot(drug2LatentFeatures, v.transpose(1, 0), dims=1)
        v = torch.tensordot(seLatentFeatures, v.transpose(2, 0), dims=1)
        v = v.transpose(0, 1).transpose(1, 2)

        return v, drugEmbeddings

    def forward(self, drug1Indices, drug2Indices, seIndices):
        drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures, drugEmbeddings = self.encode1(drug1Indices,
                                                                                                  drug2Indices,
                                                                                                  seIndices)
        scores = self.decode1(drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures)
        return scores, drugEmbeddings

    def forward2(self, drug1Indices, drug2Indices, seIndices, edge_index, edge_types):
        allLatentFeatures = self.encode2(edge_index, edge_types)
        scores, _ = self.decode2(allLatentFeatures, drug1Indices, drug2Indices, seIndices)
        return scores, allLatentFeatures


class SPARSEModel:
    r"""
    SPARSE with a fitting (training) function
    Stochastic method is applied that we only sample a sub-tensor for each iteration
    """
    def __init__(self, shape, latentSize, features, device=torch.device('cpu'), latentSizeList=None, nLayers=2):

        self.name = "SPARSE"
        self.shape = shape
        numDrug1, numDrug2, numSe = shape
        assert numDrug1 == numDrug2
        self.numDrug = numDrug1
        self.numSe = numSe
        self.device = device

        self.model = SPARSECore(self.numDrug, self.numSe, latentSize=latentSize, features=features, device=self.device,
                                n_layer=nLayers,
                                latentSizeList=latentSizeList)
        self.dim1DrugIndices = [i for i in range(self.numDrug)]
        self.dim2DrugIndices = [i for i in range(self.numDrug)]
        self.dimSeIndices = [i for i in range(self.numSe)]

    def sampleIndices(self, nSample=-1, isFull=False):
        if isFull:
            return torch.from_numpy(np.arange(0, self.numDrug)).long().to(self.device), \
                   torch.from_numpy(np.arange(0, self.numDrug)).long().to(self.device), \
                   torch.from_numpy(np.arange(0, self.numSe)).long().to(self.device)

        return torch.from_numpy(np.random.choice(self.dim1DrugIndices, nSample, replace=False)).long().to(self.device), \
               torch.from_numpy(np.random.choice(self.dim2DrugIndices, nSample, replace=False)).long().to(self.device), \
               torch.from_numpy(np.random.choice(self.dimSeIndices, nSample, replace=False)).long().to(self.device)

    def project(self):
        self.model.project()

    def getHorseShoeTerm(self):
        return 0.5 * torch.sum(self.model.latentInteractions * self.model.latentInteractions / (
                self.model.lambdaHorseShoe * self.model.lambdaHorseShoe)) / (
                       params.Tau * params.Tau) \
               + torch.sum(torch.log(self.model.lambdaHorseShoe)) \
               + torch.sum(torch.log(1 + self.model.lambdaHorseShoe * self.model.lambdaHorseShoe))

    def getLossHorseShoe(self, target, pred, w=params.L_W):
        s = target.shape
        ar = torch.full(s, w).to(self.device)
        ar[target == 1] = 1
        e = target - pred
        e = e ** 2
        e = ar * e
        r = params.Delta * torch.sum(e) + self.getHorseShoeTerm()
        return r

    def fit(self, ddiTensor, dataWrapper=None, logger=None):
        assert ddiTensor.shape == self.shape
        optimizer = torch.optim.Adam(self.model.parameters())
        sampleSize = min(params.N_SGD, self.numDrug, self.numSe)

        allRes = []
        allValAuc = []
        latentInteractionList = []
        seEmbeddingList = []
        lambdaHorseShoeList = []
        allLatentFeatures = None
        for i in range(params.N_ITER):
            optimizer.zero_grad()
            drug1Indices, drug2Indices, seIndices = self.sampleIndices(sampleSize)
            targetScores = ddiTensor
            targetScores = targetScores[drug1Indices, :, :]
            targetScores = targetScores[:, drug2Indices, :]
            targetScores = targetScores[:, :, seIndices]
            if params.FAST_TRAINING:
                predScores, _ = self.model(drug1Indices, drug2Indices, seIndices)
            else:
                predScores, allLatentFeatures = self.model.forward2(drug1Indices, drug2Indices, seIndices,
                                                                    dataWrapper.hyperEdgeIndex,
                                                                    dataWrapper.hyperEdgeTypes)

            err = self.getLossHorseShoe(targetScores, predScores)
            if i % params.ITER_DB == 0:
                print("\r%s %s" % (i, err / (sampleSize * sampleSize * sampleSize)), end="")
            err.backward()
            optimizer.step()

            if params.NON_NEG:
                self.project()

            with torch.no_grad():
                if i > 0 and i % params.CHECKPOINT_ITER == 0 and dataWrapper is not None:
                    allScores, dds = self.fullCal(allLatentFeatures=allLatentFeatures)
                    posPred = allScores[dataWrapper.testPosIndices]
                    negPred = allScores[dataWrapper.testNegIndices]
                    posValid = allScores[dataWrapper.validPosIndices]
                    negValid = allScores[dataWrapper.testNegIndices]

                    aucTest, auprTest, erT = eval_auc_aupr(posPred, negPred)
                    aucVal, auprVal, erV = eval_auc_aupr(posValid, negValid)
                    allValAuc.append(-erV)
                    allRes.append((aucTest, auprTest, aucVal, auprVal))
                    if params.LOSS_VERBOSE:
                        logger.infoAll((i, aucTest, auprTest, aucVal, auprVal, erT, erV, posPred.shape, negPred.shape))

                    latentInteractionList.append(self.model.latentInteractions.data.cpu().detach().numpy())
                    seEmbeddingList.append((dds,
                                            self.model.seEmbeddings.weight.cpu().detach().numpy()))
                    lambdaHorseShoeList.append((self.model.lambdaHorseShoe.data.cpu().detach().numpy()))

        if params.VALIDATE:
            selectedIndices = np.argmax(allValAuc)
        else:
            selectedIndices = -1
        aucTest, auprTest, aucVal, auprVal = allRes[selectedIndices]
        if params.LOSS_VERBOSE:
            print("\n\n Re:")
            print("@Ind: ", selectedIndices)
            print("Valid: ", aucVal, auprVal)
            print("Test: ", aucTest, auprTest)
            logger.infoAll("NonNeg: %s, N_SGD: %s, Using Feature: %s,  Protein Feature: %s" % (
                params.NON_NEG, params.N_SGD, params.USING_FEATURE, params.PROTEIN_FEATURE))
            logger.infoAll(("Loss Weight: ", params.L_W))
            logger.infoAll(("Valid: ", aucVal, auprVal))
            logger.infoAll(("Test: ", aucTest, auprTest))
        latentInteraction = latentInteractionList[selectedIndices]
        drugEmbeddings, seEmbeddings = seEmbeddingList[selectedIndices]
        self.saveModel(latentInteraction, drugEmbeddings, seEmbeddings, lambdaHorseShoeList[selectedIndices])
        return aucTest, auprTest

    def fullCal(self, allLatentFeatures=None):
        with torch.no_grad():
            drug1Indices, drug2Indices, seIndices = self.sampleIndices(isFull=True)
            if params.FAST_TRAINING:
                predScores, drugLatentFeatures = self.model.forward(drug1Indices, drug2Indices, seIndices)
            else:
                assert allLatentFeatures is not None
                predScores, drugLatentFeatures = self.model.decode2(allLatentFeatures, drug1Indices, drug2Indices,
                                                                    seIndices)
            return predScores.cpu().detach().numpy(), drugLatentFeatures.cpu().detach().numpy()

    @staticmethod
    def saveModel(latentInteractions, drugEmbeddings, seEmbeddings, ldv=None):
        d1, d2, d3 = latentInteractions.shape
        latentInteractions = latentInteractions.reshape(d1, d2 * d3)
        pref = "S"
        np.savetxt("%s/%s_%s_B_%s_%s_%s.txt" % (
            params.TMP_DIR, params.D_PREF, pref, params.iFold, params.Tau, int(params.HIGH_TWOSIDES)), latentInteractions)
        np.savetxt("%s/%s_%s_D_%s_%s_%s.txt" % (
            params.TMP_DIR, params.D_PREF, pref, params.iFold, params.Tau, int(params.HIGH_TWOSIDES)), drugEmbeddings)
        np.savetxt("%s/%s_%s_S_%s_%s_%s.txt" % (
            params.TMP_DIR, params.D_PREF, pref, params.iFold, params.Tau, int(params.HIGH_TWOSIDES)), seEmbeddings)
        if ldv is not None:
            ldv = ldv.reshape(d1, d2 * d3)
            np.savetxt("%s/%s_LB_%s_%s_%s.txt" % (
                params.TMP_DIR, pref, params.iFold, params.Tau, int(params.HIGH_TWOSIDES)), ldv)


def eval_auc_aupr(pos, neg):
    pos = pos.reshape(-1)
    neg = neg.reshape(-1)

    nPos, nNeg = len(pos), len(neg)
    ar = np.zeros(nPos + nNeg)
    ar[:nPos] = 1
    pred = np.concatenate((pos, neg))
    mask = np.ones((nPos + nNeg))
    mask[nPos:] = params.L_W
    er = (ar - pred) * mask
    auc, aupr = roc_auc_score(ar, pred), average_precision_score(ar, pred)
    return auc, aupr, np.mean(er * er)
