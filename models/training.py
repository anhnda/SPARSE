import numpy as np
import torch

import params
import utils.utils
from dataFactory.dataLoader import DataLoader
from models.sparse import SPARSEModel
from utils.logger.logger2 import MyLogger


def runTraining():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    allAUC = []
    allAUPR = []
    folds = [i for i in range(params.K_FOLD)]
    if params.HIGH_TWOSIDES:
        folds = [0]
    path = "%s/logs/KFolds_%s_%s_%s" % (
        params.C_DIR, params.D_PREF, params.Tau, utils.utils.getCurrentTimeString())
    logger = MyLogger(path)
    logger.infoAll(("Logging path", path))
    for iFold in folds:
        params.iFold = iFold
        wrapper = DataLoader()
        wrapper.loadData(iFold, dataPref=params.D_PREF)
        logger.infoAll("NonNeg: %s, N_SGD: %s,  Protein Feature: %s" % (
            params.NON_NEG, params.N_SGD, params.PROTEIN_FEATURE))
        logger.infoAll(("Loss Weight: ", params.L_W))
        logger.infoAll(("Data Pref: ", params.D_PREF))
        logger.infoAll(("Train on high quality twosides: ", params.HIGH_TWOSIDES))
        logger.infoAll(("Embedding", params.EMBEDDING_SIZE))
        logger.infoAll(("Horseshoe Tau ", params.Tau))

        shape = wrapper.ddiTensorInDevice.shape
        sparseModel = SPARSEModel(shape=shape, latentSize=params.EMBEDDING_SIZE, features=wrapper.features, device=device, nLayers=params.N_LAYER)
        auc, aupr = sparseModel.fit(wrapper.ddiTensorInDevice, wrapper, logger)

        allAUC.append(auc)
        allAUPR.append(aupr)

    if params.LOSS_VERBOSE and not params.HIGH_TWOSIDES:
        mauc, eauc = getMeanSE(allAUC)
        maupr, eaupr = getMeanSE(allAUPR)
        logger.infoAll(("AUC: ", mauc, eauc))
        logger.infoAll(("AUPR: ", maupr, eaupr))


def getMeanSE(ar):
    mean = np.mean(ar)
    se = np.std(ar) / np.sqrt(len(ar))
    return mean, se
