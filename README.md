# SPARSE implementation

## Preparing environment

### Create python environments using anaconda:
```shell
    conda env create -f pyv37.yml
    conda activate pyv37
    conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
    pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
    pip install torch-geometric
```

The versions of the packages tested on Ubuntu 18.04.

Other combinations of pytorch and cuda can be found at https://data.pyg.org/whl/



### Install a chrome driver (Optional for matching drugs.com)

- Download and install Google Chrome at https://www.google.com/chrome/
- Open the Chrome browser, check the version at chrome://settings/help
- Download the corresponding version of ChromeDriver at https://chromedriver.chromium.org/downloads
- Copy the ChromeDriver (chromedriver) to the bin folder of the computer. (For Linux, please copy to /usr/local/bin/)


## Commands
### Listing options

```shell
   python main.py --help
```
### Generating data for learning:

```shell
   python main.py -g -d {"" for TWOSIDES | "C" for CAD | "J" for JADER}

```

### Training SPARSE:

```shell
  python main.py -t -d {""|"C"|"J"}

```
### Extracting top predictions:

```shell
 python main.py -x 
```
The outputs are at ./tmpOut/TopPredictedTriples.txt for the top predictions and ./tmpOut/RawInterpretation.txt for interpretation with associated latent features.
### Matching with drugs.com:

```shell
 python main.py -m
```
The output is at ./tmpOut/SPARE_TopPredictions.txt for predictions matching with drugs.com.


