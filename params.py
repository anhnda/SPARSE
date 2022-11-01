r"""


This file contains global configurations and hyperparameters.

Please keep the parameters in this file by their default values.
The available options for the parameters can be found by running:

    python main.py --help


"""

import os

# Current directory
C_DIR = os.path.abspath(os.path.dirname(__file__))
# Data directory
DATA_DIR = "%s/data" % C_DIR
# Temporal directory
TMP_DIR = "%s/tmpOut" % C_DIR
# Figure directory
FIG_DIR = "%s/figs" % C_DIR
# Log directory
LOG_DIR = "%s/logs" % C_DIR

# DrugBank ATC code to Inchikey file
# Format of each line
# ATC_Code DrugName CAS_ID DrugBankID SMILE InchiKey
# e.g:
# S01EA04	Clonidine	4205-90-7	DB00575	ClC1=CC=CC(Cl)=C1NC1=NCCN1	GJSURZIOUXUGAL-UHFFFAOYSA-N
DRUGBANK_ATC_INCHI = "%s/DrugBank/ATC_2_Inchikeys.txt" % DATA_DIR

# HPRD Protein-Protein interaction file
# Format of each line:
# ProteinUniprot_1 ProteinUniport_2
PPI_FILE = "%s/HPRD/PPI_UNIPROT.txt" % DATA_DIR


# Drug Inchikey to Chemical substructures (881 Pubchem substructures) file
# Stored in the form of a (python) dictionary {inchikey: numpy.array(881))
# It is dumped by joblib
PUBCHEM_FILE = "%s/PubChem/Inchikey2ProfileFilled.dat" % DATA_DIR


# Protein-Drug binding in BingingDB database
# Format of each line
# Inchikey ProteinUniProtId
BINDINGDB_EXTENDED = "%s/DrugBank/BindingDTB" % DATA_DIR


# DrugBank Inchikey to associated proteins
# Format of each line
# InchiKey|ProteinUniprotId1,ProteinUniprotId2,...
DRUGBANK_INCHI_PROTEIN = "%s/DrugBank/DRUBBANK_INCHIKEY_2_PROTEIN.dat" % DATA_DIR


# Protein Uniprot to corresponding pathways in KEGG
# Format of each line:
# ProteinUniProtId|KEGG_PathwayName_1,KEGG_PathwayName_2,...
KEGG_UNIPROT_PATHWAY_MAPPING_PATH = "%s/KEGG/uniprot_2_pathway.txt" % DATA_DIR

# Dictionary from SMILE representation to graph representation of chemical structures.
# It is a dictionary from SMILE 2 Graph data, dumped by joblib
# This data is used for atom based model (baselines)
SMILE2GRAPH = "%s/DrugBank/SMILDE2GRAPH.dat" % DATA_DIR

# Raw response from drugs.com for checking drug-drug interactions
DRUGSCOM_DRUG_ID_RAW = "%s/DRUGSCOM/RawDrugWebText.dat" % DATA_DIR
# Drug name to Drugs.com id
DRUGSCOM_DRUG_ID_WEB = "%s/DRUGSCOM/DrugWebId.txt" % DATA_DIR

# Datasets for DDIs
DDI_DIR = "%s/DDI" % DATA_DIR

# Format of each DDI dataset:
# Each line:
# DrugName1|DrugName2|DrugInchiKey1|DrugInchiKey2|Side_effect_1,Side_effect_2,....

# Full TWOSIDES data (commonly used in related work),
# use all extracted DDIs with confident scores from 1-5.
PATH_TWOSIDES_A = "%s/TWOSIDES_ALL.txt" % DDI_DIR

# TWOSIDES data with (the highest) confidence level of 5 (a subset of full TWOSIDES)
# Use to predict more reliable interactions
PATH_TWOSIDES_C5 = "%s/TWOSIDES_C5.txt" % DDI_DIR

# DDI data for Japanese
PATH_JADERDDI = "%s/JADERDDI.txt" % DDI_DIR

# DDI data for Canadian
PATH_CADDDI = "%s/CADDDI.txt" % DDI_DIR

# Side effects of single drug
MONO_ADR_FILE = "%s/MONOADR.txt" % DDI_DIR

# Map from ID to name
ID2NamePath_TWOSIDEC5 = "%s/TWOSIDES/Id2NameC5" % TMP_DIR
EMBEDDING_PREX = "%s/Embeding_" % TMP_DIR
TORCH_SEED = int('1100110011010100111011011010011', 2)

# Optimaization method for gradient descent
OPTIMIZER = "Adam"
# Number of folds
K_FOLD = 20

# Used for debugging
ITER_DB = 10
PRINT_DB = True
INFO_OUTPUT = True
LOSS_VERBOSE = False
CHECKPOINT_ITER = 5000

# Threshold for maximum numbers of side effects (ADRs) and drugs on the filtered data.
MAX_R_ADR = 1000
MAX_R_DRUG = 600
ADR_OFFSET = 0

# Flags for using drug features
# Use protein feature
USING_FEATURE = True
PROTEIN_FEATURE = True
# Use pathway features
PATHWAY = True
# Use one-hot feature for drugs
ONE_HOT = False
# Sampling flag
KSPACE = False

# Flags for training
FAST_TRAINING = True

# Flag for using high quality TWOSIDE_C5
HIGH_TWOSIDES = False
# Validation mode (Set False to use the last epoch)
VALIDATE = False


# Prefix indicates current dataset
D_PREF = ""

# Fold Id
iFold = 0
# Size of subtensor for sampling with stochastic gradient descent
N_SGD = 80
# Maximum iterations
N_ITER = 100001
# Use non-negative embedding
NON_NEG = True
# Number of layers
N_LAYER = 2
# Embedding size
EMBEDDING_SIZE = 50

# Reweight for negative samples
L_W = 1

# Tau for Horseshoe prior
Tau = 0.02
# Delta for variant (Factor in loss function)
Delta = 1
