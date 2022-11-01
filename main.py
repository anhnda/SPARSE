import params
from models import training
from optparse import OptionParser
from dataFactory.genData import genData


def parseConfig(opts):
    r"""
        Assign values from opts to params file
    Args:
        opts: options passing from command lines

    """
    # Horseshoe tau
    params.Tau = opts.tau

    # Latent dimension size
    params.EMBEDDING_SIZE = opts.emb

    # Number of layer
    params.N_LAYER = opts.layer

    # Data prefix
    params.D_PREF = opts.data

    # Maximum number of iteration
    params.N_ITER = opts.iter
    # Flag for using high quality TWOSIDES (See params.py for detail)
    if opts.full:
        params.HIGH_TWOSIDES = True
    else:
        params.HIGH_TWOSIDES = False


def ensureDirs():
    r"""
        Create necessary folders if not exist
    """

    from utils import utils
    utils.ensure_dir(params.TMP_DIR)
    utils.ensure_dir(params.LOG_DIR)


def cleanTmp():
    r"""

    Clean temporal file

    """

    import os
    cmd = "rm -r %s/*" % params.TMP_DIR
    os.system(cmd)


if __name__ == "__main__":
    parser = OptionParser()
    # Add configuration for passing arguments
    # Horseshoe tau parameter
    parser.add_option("-t", "--tau", dest="tau", type='float', default=0.02, help="global sparsity hyperparameter")
    # Flag for Cleaning tmp file option
    parser.add_option("-c", "--clean", dest="clean", action="store_true", help="clean the tmp files")
    # Data prefix, either '' for TWOSIDES, 'C' for CADDDI, or 'C'for JADERDDI"
    parser.add_option("-d", "--data", dest="data", type='str', default="",
                      help="data prefix, either '' for TWOSIDES, 'C' for CADDDI, or 'C'for JADERDDI")
    # Flag for Generating k-fold data
    parser.add_option("-g", "--gen", dest="gen", action="store_true", help='set to generate k-fold data')
    # Flag for Training model
    parser.add_option("-r", "--train", dest="train", action="store_true", help='set to train the models')
    # Embedding size
    parser.add_option("-e", "--emb", dest="emb", type='int', default=50, help='embedding size')
    # Flag for full training on highquality TWOSIDES (TWOSIDES_C5)
    parser.add_option("-f", "--full", dest="full", action="store_true", help='set to use full TWOSIDES')
    # Flag for Extracting top predictions
    parser.add_option("-x", "--extract", dest="extract", action="store_true", help='set to extract the top predictions')
    # Number of layers
    parser.add_option("-l", "--layer", dest="layer", type='int', default=2, help='numbers of layers')
    # Maximum number of iterations
    parser.add_option("-i", "--iter", dest="iter", type='int', default=100000, help='number of iterations')
    # Flag for Matching top predictions with drugs.com
    parser.add_option("-m", "--matching", dest="matching", action="store_true",
                      help='mathching predictions with drugs.com')

    # Parse the input options
    (options, args) = parser.parse_args()
    parseConfig(options)

    print(options)
    # Create necessary folders
    ensureDirs()

    if options.gen:
        # Generate K-Fold data
        print("Generating data...")
        genData.genDataByPref(options.data)
        exit(-1)
    elif options.clean:
        # Cleaning tmp files
        print("Cleaning tmp folders")
        cleanTmp()
        exit(-1)
    elif options.train:
        # Training
        if options.full:
            # Train on full TWOSIDES_C5 without spliting into train-test.
            if options.data != "":
                print("Only support training for full high quality TWOSIDES")
                exit(-1)
        training.runTraining()
    elif options.extract:
        # Extracting new DDIs on TWOSIDE_C5
        # Require training on full TWOSIDES_C5 first
        from postProcessing.extractingTopPrediction import extract, checkModel

        options.full = True
        options.data = ""
        parseConfig(options)

        # Check whether the model is trained on TWOSIDE_C5
        isTrained = checkModel()
        if not isTrained:
            print("Retrain full TWOSIDE...")
            training.runTraining()

        print("Extracting...")
        # Extract top predictions
        extract(options.tau)


    elif options.matching:
        from postProcessing.drugsComMatching import matching
        from postProcessing.extractingTopPrediction import rematching

        # Match the predictions with drugs.com
        options.full = True
        options.data = ""

        parseConfig(options)
        # Search on drugs.com to find matching predicted drug pairs in each triple from SPARSE
        matching()
        # Merge response result from drugs.com response with the prediction result from SPARSE
        rematching(options.tau)

