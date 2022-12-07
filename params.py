r"""


Please keep the parameters in this file by their default values.
The available options for the parameters can be found by running:

    python main.py --help


"""


import os

C_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = "%s/data" % C_DIR
TMP_DIR = "%s/tmpOut" % C_DIR
FIG_DIR = "%s/figs" % C_DIR
LOG_DIR = "%s/logs" % C_DIR

DRUGBANK_ATC_INCHI = "%s/DrugBank/ATC_2_Inchikeys.txt" % DATA_DIR
PPI_FILE = "%s/HPRD/PPI_UNIPROT.txt" % DATA_DIR
PUBCHEM_FILE = "%s/PubChem/Inchikey2ProfileFilled.dat" % DATA_DIR
BINDINGDB_EXTENDED = "%s/DrugBank/BindingDTB" % DATA_DIR
DRUGBANK_INCHI_PROTEIN = "%s/DrugBank/DRUBBANK_INCHIKEY_2_PROTEIN.dat" % DATA_DIR
KEGG_UNIPROT_PATHWAY_MAPPING_PATH = "%s/KEGG/uniprot_2_pathway.txt" % DATA_DIR
SMILE2GRAPH = "%s/DrugBank/SMILDE2GRAPH.dat" % DATA_DIR

DRUGSCOM_DRUG_ID_RAW = "%s/DRUGSCOM/RawDrugWebText.dat" % DATA_DIR
DRUGSCOM_DRUG_ID_WEB = "%s/DRUGSCOM/DrugWebId.txt" % DATA_DIR
NEW_DRUGBANK_X = "%s/DrugBank/DrugBankNameX.txt" % DATA_DIR

SELECTED_DRUGS_INPUT = ""
SELECTED_DRUGS_FILTERED = ""
DDI_DIR = "%s/DDI" % DATA_DIR
PATH_TWOSIDES_A = "%s/TWOSIDES_ALL.txt" % DDI_DIR
PATH_TWOSIDES_C5 = "%s/TWOSIDES_C5.txt" % DDI_DIR
PATH_JADERDDI = "%s/JADERDDI.txt" % DDI_DIR
PATH_CADDDI = "%s/CADDDI.txt" % DDI_DIR
MONO_ADR_FILE = "%s/MONOADR.txt" % DDI_DIR

ID2NamePath_TWOSIDEC5 = "%s/TWOSIDES/Id2NameC5" % TMP_DIR
EMBEDDING_PREX = "%s/Embeding_" % TMP_DIR
TORCH_SEED = int('1100110011010100111011011010011', 2)

OPTIMIZER = "Adam"
K_FOLD = 20

ITER_DB = 10


PRINT_DB = True


MAX_R_ADR = 1000
MAX_R_DRUG = 600
ADR_OFFSET = 0

INFO_OUTPUT = True
LOSS_VERBOSE = False

PROTEIN_FEATURE = True
USING_FEATURE = True
PATHWAY = True
ONE_HOT = False
KSPACE = False


D_PREF = ""
FAST_TRAINING = True
HIGH_TWOSIDES = False
CHECKPOINT_ITER = 5000
VALIDATE = False
iFold = 0
N_SGD = 80
N_ITER = 100001
NON_NEG = True
N_LAYER = 2
EMBEDDING_SIZE = 50

L_W = 1
Tau = 0.02
Delta = 1



