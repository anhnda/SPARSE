import params
from models import training
from optparse import OptionParser
from dataFactory.genData import genData


def parseConfig(opts):
    params.Tau = opts.tau
    params.EMBEDDING_SIZE = opts.emb
    params.N_LAYER = opts.layer
    params.D_PREF = opts.data
    params.N_ITER = opts.iter
    if opts.full:
        params.HIGH_TWOSIDES = True
    else:
        params.HIGH_TWOSIDES = False


def ensureDirs():
    from utils import utils
    utils.ensure_dir(params.TMP_DIR)
    utils.ensure_dir(params.LOG_DIR)


def cleanTmp():
    import os
    cmd = "rm -r %s/*" % params.TMP_DIR
    os.system(cmd)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-t", "--tau", dest="tau", type='float', default=0.02, help="global sparsity hyperparameter")
    parser.add_option("-c", "--clean", dest="clean", action="store_true", help="clean the tmp files")
    parser.add_option("-d", "--data", dest="data", type='str', default="", help="data prefix, either '' for TWOSIDES, 'C' for CADDDI, or 'C'for JADERDDI")
    parser.add_option("-g", "--gen", dest="gen", action="store_true", help='set to generate k-fold data')
    parser.add_option("-r", "--train", dest="train", action="store_true", help='set to train the models')
    parser.add_option("-e", "--emb", dest="emb", type='int', default=50, help='embedding size')
    parser.add_option("-f", "--full", dest="full", action="store_true", help='set to use full TWOSIDES')
    parser.add_option("-x", "--extract", dest="extract", action="store_true", help='set to extract the top predictions')
    parser.add_option("-l", "--layer", dest="layer", type='int', default=2, help='numbers of layers')
    parser.add_option("-i", "--iter", dest="iter", type='int', default=100000, help='number of iterations')
    parser.add_option("-m", "--matching", dest="matching", action="store_true", help='mathching predictions with drugs.com')

    (options, args) = parser.parse_args()
    parseConfig(options)

    print(options)
    ensureDirs()

    if options.gen:
        print("Generating data...")
        genData.genDataByPref(options.data)
        exit(-1)
    elif options.clean:
        print("Cleaning tmp folders")
        cleanTmp()
        exit(-1)
    elif options.train:
        if options.full:
            if options.pref != "":
                print("Only support training for full high quality TWOSIDES")
                exit(-1)
        training.runTraining()
    elif options.extract:
        # Require training on full TWOSIDES first
        from postProcessing.extractingTopPrediction import extract, checkModel

        options.full = True
        options.pref = ""
        parseConfig(options)
        isTrained = checkModel()
        if not isTrained:
            print("Retrain full TWOSIDE...")
            training.runTraining()

        print("Extracting...")
        extract(options.tau)


    elif options.matching:
        from postProcessing.drugsComMatching import matching
        from postProcessing.extractingTopPrediction import rematching
        options.full = True
        options.pref = ""
        parseConfig(options)
        matching()
        rematching(options.tau)
