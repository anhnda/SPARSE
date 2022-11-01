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
        r"""
        Init SPARSECore module with forward functions (consisting of encoders and decoders)
        Args:
            numDrug: Number of drug
            numSE: Number of side effect
            latentSize: Latent feature size
            features: Drug feature matrix
            n_layer: Number of layer
            device: device : cpu or cuda (gpu)
            latentSizeList: list of latent feature sizes (in cases with different latent features sizes for drug and side effect
        """
        super(SPARSECore, self).__init__()
        # Assign the number of drugs, side effects, size of latent features, device, nlayers

        self.numDrug = numDrug
        self.numSE = numSE
        self.latentSize = latentSize
        self.device = device
        self.n_layer = n_layer

        # Assign the latent feature size for drug and side effects
        latentSize1 = latentSize
        latentSize2 = latentSize
        # In cases the sizes of latent feature sizes are different
        if latentSizeList is not None and len(latentSizeList) > 0:
            latentSize1, latentSize2 = latentSizeList
            self.latentSize = latentSize1

        # Two layers for transforming input drug features to latent spaces
        self.neuralList = torch.nn.ModuleList()
        self.neuralList.append(torch.nn.Linear(features.shape[-1], 2 * latentSize1).to(device))
        self.neuralList.append(torch.nn.Linear(2 * latentSize1, latentSize1).to(device))
        self.neuralAct = torch.nn.ReLU()

        # Add convolutional layers
        self.convs = torch.nn.ModuleList()
        self.convAct = torch.nn.ReLU()
        for i in range(self.n_layer):
            layer = HyperConv(latentSize, latentSize, 5, device=device)
            self.convs.append(layer)

        # Embedding tables for side effects
        self.seEmbeddings = torch.nn.Embedding(numSE, latentSize2).to(device)
        # Indices for side effects
        self.seIndices = torch.arange(0, numSE, dtype=torch.long).to(device)

        # Latent interactions (Tensor B)
        self.latentInteractions = torch.nn.Parameter(
            torch.rand((latentSize1, latentSize1, latentSize2), requires_grad=True).to(device))

        # Init tensors
        torch.nn.init.xavier_uniform_(self.seEmbeddings.weight)
        torch.nn.init.xavier_uniform_(self.latentInteractions.data)

        # Assign drug features
        self.drugFeatures = features
        # Horseshoe prior: lambda
        lambdaHorseShoe = halfcauchy.rvs(size=latentSize1 * latentSize1 * latentSize2)
        lambdaHorseShoe = lambdaHorseShoe.reshape((latentSize1, latentSize1, latentSize2))
        self.lambdaHorseShoe = torch.nn.Parameter(torch.from_numpy(lambdaHorseShoe).float(), requires_grad=True).to(
            device)

    def project(self):
        r"""
        Non-negative projection for latent interactions, and latent features


        """
        self.latentInteractions.data[self.latentInteractions.data < 0] = 0
        self.lambdaHorseShoe.data[self.lambdaHorseShoe.data < 0] = 0
        self.seEmbeddings.weight.data[self.seEmbeddings.weight.data < 0] = 0

    def encode1(self, drug1Indices, drug2Indices, seIndices):
        r"""
        # Encode input to drug latent features, side effect latent features
        # encode1 fixes the convolutional layer as an identical function
        # This speeds up the training process and make the training more stable.
        Args:
            drug1Indices: Indices of drug1 [drugIndex_{1,1}, drugIndex{1,2}, ..., drugIndex{1,n}]
            drug2Indices: Indices of drug2 [drugIndex_{2,1}, drugIndex{2,2}, ..., drugIndex{2,n}]
            seIndices: Indices of se [SeIndex_{1}, SeIndex_{2}, ..., SeIndex_{n}]

        Returns:
            drug1LatentFeatures: Latent features of drugs in drug1Indices
            drug2LatentFeatures: Latent features of drugs in drug2Indices
            seLatentFeatures: Latent features of side effects in seIndices
            drugLatentFeatures: Latent features of all drugs
        """

        # Get all drug input features
        drugLatentFeatures = self.drugFeatures
        # Transforming all drug input features through 2 layer feedforward neural network
        for nnLayer in self.neuralList:
            drugLatentFeatures = self.neuralAct(nnLayer(drugLatentFeatures))

        # Get corresponding drug latent features of drug1 and drug2
        drug1LatentFeatures = drugLatentFeatures[drug1Indices]
        drug2LatentFeatures = drugLatentFeatures[drug2Indices]

        # Get Se embeddings
        seLatentFeatures = self.seEmbeddings(seIndices)
        return drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures, drugLatentFeatures

    def decode1(self, drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures):
        r"""
        Reconstructed DDI scores given latent features of drug1, drug2, and side effect

        # Use the tensor product in Equation 17 in the SPARSE paper to reconstruct DDIs
        # V = LatentInteraction (B) x_1 DrugLatentFeature1 x_2 DrugLatentFeature2 x_3 SELatentFeatures
        Args:
            drug1LatentFeatures:
            drug2LatentFeatures:
            seLatentFeatures:

        Returns:
            subTensor for DDIs of given drugs and side effects

        """
        # Use tensor product Equation to reconstruct DDIs
        # V = LatentInteraction (B) x_1 DrugLatentFeature1 x_2 DrugLatentFeature2 x_3 SELatentFeatures
        v = torch.tensordot(drug1LatentFeatures, self.latentInteractions, dims=1)
        v = torch.tensordot(drug2LatentFeatures, v.transpose(1, 0), dims=1)
        v = torch.tensordot(seLatentFeatures, v.transpose(2, 0), dims=1)
        v = v.transpose(0, 1).transpose(1, 2)
        return v

    def encode2(self, edgeIndex, edgeTypes):
        r"""
        # Encode input to drug latent features, side effects effect latent features
        # Encode2 use convolutional layers.
        # In theory, by learning parameter from convolutional layers,
        # It is expected to find a better solution than an identical function
        # However, it is not guarantee that we can find a better one in practice
        # Also the training process is much slower

        # Recommend to use Encode1

        Args:
            edgeIndex: Edges of drug-drug, drug-side effects which are clique expansions of hyperedges,
                      eg: (d1, d2, e) -> (d1, d2), (d1, e), (d2,e) (also with self-loop (d1,d1), (d2,d2), (e,e))
            edgeTypes: Corresponding types of edge (drug-drug, drug-se, se-drug, or self-loop) (c(a), c(b) in Eq. 15)

        Returns:
         all latent features of drugs and side effects
        """
        # Get drug latent features
        drugLatentFeatures = self.drugFeatures
        for i, layer in enumerate(self.neuralList):
            drugLatentFeatures = layer(drugLatentFeatures)
            if i < len(self.neuralList) - 1:
                drugLatentFeatures = self.neuralAct(drugLatentFeatures)
        # Get side effect latent features
        seLatentFeatures = self.seEmbeddings(self.seIndices)
        # Do convolution
        allLatentFeatures = torch.vstack((drugLatentFeatures, seLatentFeatures))
        for layer in self.convs:
            allLatentFeatures = self.convAct(layer(allLatentFeatures, edgeIndex, edgeTypes))
        return allLatentFeatures

    def decode2(self, allLatentFeatures, drug1Indices, drug2Indices, seIndices):
        r"""
        # Reconstructing DDIs from latent features and selected indices

        Args:
            allLatentFeatures: all latent features of drugs and side effects
            drug1Indices: indices of drug1
            drug2Indices: indices of drug2
            seIndices: indices of side effect

        Returns:
            predicted sub DDI tensors, drug embeddings
        """
        # Get latent features
        drugEmbeddings = allLatentFeatures[:self.numDrug, ]
        seEmbeddings = allLatentFeatures[self.numDrug:, ]
        drug1LatentFeatures = drugEmbeddings[drug1Indices]
        drug2LatentFeatures = drugEmbeddings[drug2Indices]
        seLatentFeatures = seEmbeddings[seIndices]
        # Calculate tensor product:
        # V = LatentInteraction (B) x_1 DrugLatentFeature1 x_2 DrugLatentFeature2 x_3 SELatentFeatures

        v = torch.tensordot(drug1LatentFeatures, self.latentInteractions, dims=1)
        v = torch.tensordot(drug2LatentFeatures, v.transpose(1, 0), dims=1)
        v = torch.tensordot(seLatentFeatures, v.transpose(2, 0), dims=1)
        v = v.transpose(0, 1).transpose(1, 2)

        return v, drugEmbeddings

    def forward(self, drug1Indices, drug2Indices, seIndices):
        r"""
             Reconstructing DDIs with encode1, decode1

        Args:
            drug1Indices: Indices of drug1
            drug2Indices: Indices of drug2
            seIndices: Indices of side effects

        Returns: sub-Tensor for reconstructed DDI scores of given drug indices and side effect indices

        """
        drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures, drugEmbeddings = self.encode1(drug1Indices,
                                                                                                  drug2Indices,
                                                                                                  seIndices)
        scores = self.decode1(drug1LatentFeatures, drug2LatentFeatures, seLatentFeatures)
        return scores, drugEmbeddings

    def forward2(self, drug1Indices, drug2Indices, seIndices, edge_index, edge_types):
        r"""
        # Reconstructing DDIs with encode2, decode2

        Args:
            drug1Indices: Indices of drug1
            drug2Indices: Indices of drug2
            seIndices: Indices of side effects
            edge_index: Indices of edge for clique expansion of hyperedges (Eq. 13-15 in the paper)
            edge_types: Corresponding edge type (( c(a), c(b) Eq. 15 in the paper)

        Returns:

        """
        allLatentFeatures = self.encode2(edge_index, edge_types)
        scores, _ = self.decode2(allLatentFeatures, drug1Indices, drug2Indices, seIndices)
        return scores, allLatentFeatures


class SPARSEModel:
    r"""
    SPARSE with a fitting (training) function
    Stochastic method is applied that we only sample a sub-tensor for each iteration
    """
    def __init__(self, shape, latentSize, features, device=torch.device('cpu'), latentSizeList=None, nLayers=2):
        r"""
        Init SPARSE model from SPARSECore
        Args:
            shape: DDI tensor shape: |V_D| x |V_D| x |V_S|
            latentSize: latent feature size
            features: drug feature matrix
            device: cpu or cuda (gpu)
            latentSizeList: list of latent feature sizes
            nLayers: number of layers

        """
        # Assign shape and device
        self.name = "SPARSE"
        self.shape = shape
        numDrug1, numDrug2, numSe = shape
        assert numDrug1 == numDrug2
        self.numDrug = numDrug1
        self.numSe = numSe
        self.device = device
        # Init SPARSECore
        self.model = SPARSECore(self.numDrug, self.numSe, latentSize=latentSize, features=features, device=self.device,
                                n_layer=nLayers,
                                latentSizeList=latentSizeList)
        # All indices for drug1, drug2, se effects
        # Name convention: each triple (hyperedge): (drug1, drug2, side effect)
        # drug1 refers to the first drug in a case of one triple,
        # or a list the first drugs in cases of a list of triples
        # Similar for drug2

        self.dim1DrugIndices = [i for i in range(self.numDrug)]
        self.dim2DrugIndices = [i for i in range(self.numDrug)]
        self.dimSeIndices = [i for i in range(self.numSe)]

    def sampleIndices(self, nSample=-1, isFull=False):
        r"""
        Sample a sub-tensor of size nSample x nSample x nSample from the DDI tensor of numDrug x numDrug x numSE
        Args:
            nSample: sample size
            isFull: return the indices of the whole DDI tensor

        Returns:
            the sampled dimensions of the sub-tensors

        """

        if isFull:
            # Get all indices of drug1, drug2, side effect
            return torch.from_numpy(np.arange(0, self.numDrug)).long().to(self.device), \
                   torch.from_numpy(np.arange(0, self.numDrug)).long().to(self.device), \
                   torch.from_numpy(np.arange(0, self.numSe)).long().to(self.device)
        else:
            # Sample some indices of drug1, some indices of drug2, and side effects
            return torch.from_numpy(np.random.choice(self.dim1DrugIndices, nSample, replace=False)).long().to(self.device), \
                   torch.from_numpy(np.random.choice(self.dim2DrugIndices, nSample, replace=False)).long().to(self.device), \
                   torch.from_numpy(np.random.choice(self.dimSeIndices, nSample, replace=False)).long().to(self.device)

    def project(self):
        # For non-negative projection
        self.model.project()

    def getHorseShoeTerm(self):
        r"""
        Horseshoe term in Equation 20-21 on the paper
        """
        return 0.5 * torch.sum(self.model.latentInteractions * self.model.latentInteractions / (
                self.model.lambdaHorseShoe * self.model.lambdaHorseShoe)) / (
                       params.Tau * params.Tau) \
               + torch.sum(torch.log(self.model.lambdaHorseShoe)) \
               + torch.sum(torch.log(1 + self.model.lambdaHorseShoe * self.model.lambdaHorseShoe))

    def getLossHorseShoe(self, target, pred, w=params.L_W):
        r"""
            The loss for the objective function in Eq. 19

        Args:
            target: Target DDI sub-tensor
            pred: Predicted DDI sub-tensor
            w: weight for positive and negative DDIs

        Returns:
            loss for optimizing

        """

        s = target.shape
        ar = torch.full(s, w).to(self.device)
        ar[target == 1] = 1
        e = target - pred
        e = e ** 2
        e = ar * e
        r = params.Delta * torch.sum(e) + self.getHorseShoeTerm()
        return r

    def fit(self, ddiTensor, dataWrapper=None, logger=None):
        r"""
        fit the SPARSE model with the input DDI tensor and data of the given fold

        Args:
            ddiTensor: input DDI tensor of numDrug x numDrug x numSE
            dataWrapper: wrapper data for DDI
            logger: logger object

        Returns: auc, aupr
        """

        assert ddiTensor.shape == self.shape
        # Optimizer with learnable parameters
        optimizer = torch.optim.Adam(self.model.parameters())

        # Get sample size in each dimension (of the tensor |V_D| x |V_D| x |V_S|)
        sampleSize = min(params.N_SGD, self.numDrug, self.numSe)
        # List of all results at checkpoints
        allRes = []
        # List of all auc at checkpoints
        allValAuc = []
        # List of all latent interactions at checkpoints
        latentInteractionList = []
        # List of all embedding at checkpoints
        seEmbeddingList = []
        # List of all horseshoe prior at checkpoints
        lambdaHorseShoeList = []

        allLatentFeatures = None
        for i in range(params.N_ITER):
            # Reset gradient
            optimizer.zero_grad()
            # Sample indices for a sub-tensor from DDI tensor
            drug1Indices, drug2Indices, seIndices = self.sampleIndices(sampleSize)

            # Get target scores for the sub-tensor
            targetScores = ddiTensor
            targetScores = targetScores[drug1Indices, :, :]
            targetScores = targetScores[:, drug2Indices, :]
            targetScores = targetScores[:, :, seIndices]
            # Use the default mode with fast-training: encode1-decode1
            if params.FAST_TRAINING:
                predScores, _ = self.model(drug1Indices, drug2Indices, seIndices)
            else:
                predScores, allLatentFeatures = self.model.forward2(drug1Indices, drug2Indices, seIndices,
                                                                    dataWrapper.hyperEdgeIndex,
                                                                    dataWrapper.hyperEdgeTypes)
            # Get loss with horseshoe regularization
            err = self.getLossHorseShoe(targetScores, predScores)
            if i % params.ITER_DB == 0:
                print("\r%s %s" % (i, err / (sampleSize * sampleSize * sampleSize)), end="")
            # Backward and update
            err.backward()
            optimizer.step()

            # Projecting to keep non-negative constraints
            if params.NON_NEG:
                self.project()

            with torch.no_grad():
                # Calculate predictions at checkpoints
                if i > 0 and i % params.CHECKPOINT_ITER == 0 and dataWrapper is not None:
                    # All DDIs scores
                    allScores, dds = self.fullCal(allLatentFeatures=allLatentFeatures)

                    # Extract corresponding scores on the holdout validation-test set
                    posPred = allScores[dataWrapper.testPosIndices]
                    negPred = allScores[dataWrapper.testNegIndices]
                    posValid = allScores[dataWrapper.validPosIndices]
                    negValid = allScores[dataWrapper.testNegIndices]

                    # Calculate AUC-AUPR
                    aucTest, auprTest, erT = eval_auc_aupr(posPred, negPred)
                    aucVal, auprVal, erV = eval_auc_aupr(posValid, negValid)

                    # Save results to lists
                    allValAuc.append(-erV)
                    allRes.append((aucTest, auprTest, aucVal, auprVal))

                    if params.LOSS_VERBOSE:
                        logger.infoAll((i, aucTest, auprTest, aucVal, auprVal, erT, erV, posPred.shape, negPred.shape))

                    latentInteractionList.append(self.model.latentInteractions.data.cpu().detach().numpy())
                    seEmbeddingList.append((dds,
                                            self.model.seEmbeddings.weight.cpu().detach().numpy()))
                    lambdaHorseShoeList.append((self.model.lambdaHorseShoe.data.cpu().detach().numpy()))
        # If use validation: best auc
        if params.VALIDATE:
            selectedIndices = np.argmax(allValAuc)
        else:
        # By default, use the last model
            selectedIndices = -1

        # Retrieve the selected result
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

        # Save learned model
        latentInteraction = latentInteractionList[selectedIndices]
        drugEmbeddings, seEmbeddings = seEmbeddingList[selectedIndices]
        self.saveModel(latentInteraction, drugEmbeddings, seEmbeddings, lambdaHorseShoeList[selectedIndices])
        # Return auc-aupr on the test set
        return aucTest, auprTest

    def fullCal(self, allLatentFeatures=None):
        # Reconstruct the whole DDI tensor |V_D| x |V_D| x |V_S|
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
        # Save learned latent features and latent interactions
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
    r"""

    Args:
        pos: Predicted scores of the positive samples
        neg: Predicted scores of the negative samples

    Returns:
        AUC, AUPR, weighted_error
    """
    # Construct the ground truth:
    pos = pos.reshape(-1)
    neg = neg.reshape(-1)

    nPos, nNeg = len(pos), len(neg)
    # Assign ground truth for negative samples with 0
    ar = np.zeros(nPos + nNeg)
    # Assign ground truth for positive sampels with 1
    ar[:nPos] = 1
    # Full prediction
    pred = np.concatenate((pos, neg))
    mask = np.ones((nPos + nNeg))
    # Reweight for positive and negative sample (By default: same weight)
    mask[nPos:] = params.L_W
    er = (ar - pred) * mask
    # Get AUC, AUPR
    auc, aupr = roc_auc_score(ar, pred), average_precision_score(ar, pred)
    return auc, aupr, np.mean(er * er)
