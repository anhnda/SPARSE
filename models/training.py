import numpy as np
import torch

import params
import utils.utils
from dataFactory.dataLoader import DataLoader
from models.sparse import SPARSEModel
from utils.logger.logger2 import MyLogger


def runTraining():
    # Get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # List of auc, aupr
    allAUC = []
    allAUPR = []
    # List of fold id
    folds = [i for i in range(params.K_FOLD)]
    # In case with full training on TWOSIDES_C5 (High quality TWOSIDES), only train on the first fold (since there is no train-test splitting)
    if params.HIGH_TWOSIDES:
        folds = [0]
    # Init logger
    path = "%s/logs/KFolds_%s_%s_%s" % (
        params.C_DIR, params.D_PREF, params.Tau, utils.utils.getCurrentTimeString())
    logger = MyLogger(path)
    logger.infoAll(("Logging path", path))


    for iFold in folds:
        # Train for fold i
        params.iFold = iFold
        # Load data for fold i
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
        # Init SPARSE
        sparseModel = SPARSEModel(shape=shape, latentSize=params.EMBEDDING_SIZE, features=wrapper.features, device=device, nLayers=params.N_LAYER)
        # Fit SPARSE model with the current fold data
        auc, aupr = sparseModel.fit(wrapper.ddiTensorInDevice, wrapper, logger)
        # Append auc, aupr to list
        allAUC.append(auc)
        allAUPR.append(aupr)

    if params.LOSS_VERBOSE and not params.HIGH_TWOSIDES:
        # Calculating mean and standard error of aucs and auprs
        mauc, eauc = getMeanSE(allAUC)
        maupr, eaupr = getMeanSE(allAUPR)
        logger.infoAll(("AUC: ", mauc, eauc))
        logger.infoAll(("AUPR: ", maupr, eaupr))


def getMeanSE(ar):
    r"""
    get mean and standard error from a given array of values
    Args:
        ar: array of values

    Returns:
        mean and starndard error

    """
    mean = np.mean(ar)
    se = np.std(ar) / np.sqrt(len(ar))
    return mean, se
