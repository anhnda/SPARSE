from selenium import webdriver
import time
from utils import utils
from bs4 import BeautifulSoup
import params
from selenium.webdriver.common.by import By

Pref = "https://www.drugs.com/js/search/?id=livesearch-interaction&s="
INTER_PREF = "https://www.drugs.com/interactions-check.php?drug_list="
RAW_DRUG_TEXT = params.DRUGSCOM_DRUG_ID_RAW
DRUG_WEB_ID_PATH = params.DRUGSCOM_DRUG_ID_WEB

PREDICTION_PATH = "%s/TopPredictedTriples.txt" % params.TMP_DIR
RAW_RES_INTER = "%s/RawDrugComResponse.dat" % params.TMP_DIR


def loadDrugList():
    r"""
    load valid drug name list
    Returns:
        A list of drug names in full high quality TWOSIDES

    """
    fin = open("%s/TWOSIDES/DrugId2NameC5.txt" % params.TMP_DIR)
    drugList = []
    for line in fin.readlines():
        ss = line.strip().split("\t")
        drugList.append(ss[1].lower())
    fin.close()
    return drugList


def getRetrieveDrugURL(drugName):
    r"""
    get URL for drugs.com query id from drug name

    Args:
        drugName: drug name

    Returns:
        An URL for getting Drugs.com Id given a drug name
    """
    return "%s%s" % (Pref, drugName)


def downloadDrugWebId():
    r"""
    Use annotation with Selenium to get raw responses from drugs.com for mapping from drug names to drug ids on drugs.com

    Note: Drugs.com uses its id for each drug.

    The result is saved in RAW_DRUG_TEXT
    """
    browser = webdriver.Chrome()

    drugList = loadDrugList()
    try:
        dDrugName2Re = utils.load_obj(RAW_DRUG_TEXT)
    except:
        dDrugName2Re = dict()
    # Iterate for drug name list
    for drug in drugList:
        # If drug name is already retrieved then skip
        if drug in dDrugName2Re:
            continue

        print("\r %s, %s" % (len(dDrugName2Re), drug), end="")
        # Get corresponding url query for drug name
        urlx = getRetrieveDrugURL(drug)
        # Get response from given url
        browser.get(urlx)
        # Extract body html
        html = browser.find_elements(By.TAG_NAME, 'body')[0]
        html = html.get_attribute('innerHTML')
        dDrugName2Re[drug] = html
        # Delay to avoid too many requests
        time.sleep(3)
        # Save the results
        if len(dDrugName2Re) % 10 == 0:
            utils.save_obj(dDrugName2Re, RAW_DRUG_TEXT)
            print(html)
    # Save all raw retrieved result for drugs.com id from drug names
    utils.save_obj(dDrugName2Re, RAW_DRUG_TEXT)


def parsex(pin=RAW_DRUG_TEXT, pout=DRUG_WEB_ID_PATH):
    r"""
    Extracting the mapping from drug names to drug ids given the raw responses from drugs.com
    Args:
        pin: The path to the raw responses
        pout: The path to the clean mapping


    """
    # Load raw response from drug name to drugs.com id
    d = utils.load_obj(pin)
    fout = open(pout, "w")
    # Iterating each drugname - rawresponse and extract drugs.com id
    # Please use inspecting function of Chrome browser to see the location of the
    # the drugs.com id on the response
    # The following extraction is to find the corresponding element of the drugs.com id
    for k, v in d.items():
        rex = []
        try:
            vbody = BeautifulSoup(v, "html.parser")
            c = vbody.find('a', {"class": "ls-item"})
            txt = c['onclick']
            i1 = txt.index('(')
            i2 = txt.index(')')
            re = txt[i1 + 1:i2]
            parts = re.split(",")

            for part in parts:
                part = part.strip()
                val = part[1:-1]
                rex.append(val)
            # if not rex[-1] == k:
            #     rex = []
        except:
            pass
        if len(rex) > 0:
            fout.write("%s||%s\n" % (k, ",".join(rex)))
    fout.close()


def getInteractions(drugWebIdPath=DRUG_WEB_ID_PATH, predictionPath=PREDICTION_PATH, pOut=RAW_RES_INTER):
    r"""
    Use annotation with Selenium to check drug-drug interactions
    Args:
        drugWebIdPath: path for the mapping from drug names to drugs.com ids
        predictionPath: path for predicted drug interactions
        pOut: path for the output file of raw responses from drugs.com

    """


    fin = open(drugWebIdPath)
    lines = fin.readlines()
    dDrugName2WebId = dict()

    browser = webdriver.Chrome()
    # Dictionary to store raw result from input drug pair to drugs.com html responses
    dDrugPairToRe = dict()
    print("Start...")
    try:
        dDrugPairToRe = utils.load_obj(pOut)
    except:
        pass
    print("Init len: ", len(dDrugPairToRe))
    # Creating pair of drugs.com id given pairs of drug name
    for line in lines:
        line = line.strip()
        parts = line.split("||")
        drugName = parts[0]
        info = parts[1].split(",")
        k1 = info[0]
        k2 = info[1]
        dDrugName2WebId[drugName] = "%s-%s" % (k1, k2)
    fin.close()

    try:
        currentRe = utils.load_obj(pOut)
    except:
        currentRe = {}
    validDrugs = dDrugName2WebId.keys()
    print("N valid drugs: ", len(validDrugs))

    # Sort drug pairs to make sure each pair is only searched one time.
    def srt(v1, v2):
        if v1 > v2:
            v1, v2 = v2, v1
        return v1, v2

    # Load the prediction triples
    fin = open(predictionPath)
    lines = fin.readlines()
    print("Loop")
    # Loo
    for line in lines:
        # print(line)
        try:
            line = line.strip().lower()
            parts = line.split(",")
            d1 = parts[0].strip()
            d2 = parts[1].strip()
            d1, d2 = srt(d1, d2)
            if d1 not in validDrugs or d2 not in validDrugs:
                continue
            p = "%s,%s" % (d1, d2)
            if p in currentRe:
                continue
            # Get pair of drugs.com id
            pair = "%s,%s" % (dDrugName2WebId[d1], dDrugName2WebId[d2])
            # Get drugs.com interaction checker url
            urlx = "%s%s" % (INTER_PREF, pair)
            print("\r %s, %s, %s" % (len(dDrugPairToRe), p, urlx), end="")
            # Get raw html reponses (body part)
            browser.get(urlx)
            html = browser.find_elements(By.TAG_NAME, 'body')[0]
            html = html.get_attribute('innerHTML')
            dDrugPairToRe[p] = html
            # Delay to avoid too many requests
            time.sleep(4)
            # Save results
            if len(dDrugPairToRe) % 10 == 0:
                utils.save_obj(dDrugPairToRe, pOut)
                print(html[:20])
        except Exception as e:
            print(e)
            continue
    # Save all results
    utils.save_obj(dDrugPairToRe, pOut)


def extractInteraction():
    r"""
    Extracting drug-drug interactions information from raw responses of drugs.com
    """
    fMatching = open("%s/PairMatching.txt" % params.TMP_DIR, "w")
    fnoMatching = open("%s/PairNoMatching.txt" % params.TMP_DIR, "w")
    d = utils.load_obj(RAW_RES_INTER)
    cc = 0
    print("N Pais: ", len(d))
    # Use inspection function on Chrome browser to see the location of the interaction section on the response
    # The following code is used to extract the corresponding interaction section
    for k, v in d.items():
        cc += 1
        vbody = BeautifulSoup(v, "html.parser")
        div = vbody.find("div", {'class': 'interactions-reference-wrapper'})
        isNoMatch = False
        try:
            if div.text.__contains__('No interactions were found between the drugs in your list.'):
                fnoMatching.write("%s\n" % k)
                isNoMatch = True
        except Exception as e:
            print("1 Error in No Matching")
            print(e)
            pass

        if not isNoMatch:
            try:
                pp = div.findAll("p")
                pr = []
                for p in pp:
                    txt = p.text
                    pr.append(txt)
                fMatching.write("%s||%s\n" % (k, ". ".join(pr).replace("\n", ". ")))
            except Exception as e:
                print("- Error In Matching")
                print(e)

    fMatching.close()
    fnoMatching.close()


def getDrugsComIds():
    r"""
    First get the raw responses from drugs.com for drug names to drug ids
    Then extract the mapping from drug names
    """
    downloadDrugWebId()
    parsex()


def matching():
    r"""
    First get the raw responses from drugs.com for drug interactions
    Then extract the interactions from the raw responses
    """
    getInteractions()
    extractInteraction()


if __name__ == "__main__":
    # getDrugsComIds()
    # getInteractions()
    # extractInteraction()
    pass
