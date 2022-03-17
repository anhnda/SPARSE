from torch_geometric.data import Data, Batch

from utils import utils
import torch
import numpy as np
import params


class MoleculeFactory:
    r"""
    MoleculeFactory is used to generate molecular graphs from SMILE representations
    """
    def __init__(self):
        self.__atomElement2Id = dict()
        self.moleculeList = list()
        self.smile2Graph = utils.load_obj(params.SMILE2GRAPH)

    def getAtomIdFromElement(self, ele):
        return utils.get_update_dict_index(self.__atomElement2Id, ele)

    def convertSMILE2Graph(self, smile):
        mol = self.smile2Graph[smile]
        nodes = mol._node
        edges = mol._adj
        nodeFeatures = []
        assert len(nodes) != 0

        keys = nodes.keys()
        keys = sorted(keys)

        mapKeys = dict()
        for k in keys:
            mapKeys[k] = len(mapKeys)

        for nodeId in keys:
            nodeDict = nodes[nodeId]
            element = nodeDict['element']
            atomId = self.getAtomIdFromElement(element)

            charger = nodeDict['charge']
            aromatic = nodeDict['aromatic']
            hcount = nodeDict['hcount']
            nodeFeature = [element, atomId, charger, aromatic, hcount]
            nodeFeatures.append(nodeFeature)

        edgeIndex = []
        edgeAttr = []

        for nodeId, nextNodes in edges.items():
            for nextNodeId, edgeInfo in nextNodes.items():
                edgeIndex.append([mapKeys[nodeId], mapKeys[nextNodeId]])
                edgeAttr.append([edgeInfo['order']])

        return [nodeFeatures, edgeIndex, edgeAttr]

    def addSMILE(self, smile):
        self.moleculeList.append(self.convertSMILE2Graph(smile))

    def getNumAtom(self):
        return len(self.__atomElement2Id)

    def createBatchGraph(self, atomOffset=0):
        self.N_ATOM = self.getNumAtom()
        self.N_FEATURE = self.N_ATOM
        graphList = list()
        cc = 0
        for modeculeInfo in self.moleculeList:
            nodeFeatures, edgIndex, edgeAttr = modeculeInfo
            nodeVecs = []
            for nodeFeature in nodeFeatures:
                element, atomId, charger, aromatic, hcount = nodeFeature
                nodeVecs.append(atomId + atomOffset)

            cc += len(nodeFeatures)
            newEdgIndex = []
            for edge in edgIndex:
                i1, i2 = edge
                newEdgIndex.append([i1, i2])

            nodeVecs = np.asarray(nodeVecs)
            nodeVecs = torch.from_numpy(nodeVecs).long()
            newEdgIndex = torch.from_numpy(np.asarray(newEdgIndex)).long().t().contiguous()

            # edgeAttr = torch.from_numpy(np.asarray(edgeAttr)).float()

            data = Data(x=nodeVecs, edge_index=newEdgIndex)
            graphList.append(data)

        self.graphList = graphList

        batch = Batch.from_data_list(graphList)
        print("Batch molecular graph completed.")
        print("Total: ", cc, len(self.moleculeList), cc * 1.0 / len(self.moleculeList))

        return batch
