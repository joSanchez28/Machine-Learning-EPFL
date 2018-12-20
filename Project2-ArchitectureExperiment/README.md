# Architecture feelings - EPFL Machine Learning

This is the second class project of the 2018 EPFL Machine Learning course.
In this document you can see an overview of the project and how to
run it, as well as some useful information about the requirements and the
implementation. If you require more specific information, you can check the
report (`report.pdf`).

## Project folders
The project has the following folder (and file) structure:

* `ArchitectureFeelings-Notebook.ipynb` Main Jupyter notebook where you can find the main development of the project. - If you want to follow the project development we really encourage you to see this notebook.
* `Modeling_every_label_demo.ipynb` Second and last Jupyter notebook. In this notebook we extrapolate our study in the 'exciting' feeling to the rest of the feelings.
* `data/`. Directory with the data sets given by the hosting lab collaborator,
namely `T_features.csv` and
`table_dataset_GreeceSwitzerland_N265_metrics_mSC5_JPEGtoBMP_Michelson_RMS.csv`; and also data sets generated, `original_data.parquet` and `pixels_compressed.parquet`.
    * `BMP/` Directory containing the folders with the different images given by the hosting lab collaborator. For each scene we have the cube1,...,cube6 and cube_persp images.
* `docs/`. Directory with the report in pdf `report.pdf` and in `report.tex`.
* `scripts/` Python files
    * `ArchitectureFeelings.py`. Just in case you can not (or you do not want to) run the code of `ArchitectureFeelings-Notebook.ipynb` with the Jupyter notebook; you can run this as an usual python script.
    * `Modeling_every_label_demo.py`. Just in case you can not (or you do not want to) run the code of `Modeling_every_label_demo.ipynb`
* `visualizations/` Directory containing the images (plots and visualizations) saved in during the notebooks execution.
* `sklearn_models/` Directory with the most of the grid searches done in both notebooks and saved as `.pkl` with the joblib library.


## Requirements for running the project
We recommend the [Anaconda environment](https://www.anaconda.com/download/) with Jupyter notebook for open and running both notebooks, `ArchitectureFeelings-Notebook.ipynb` and `Modeling_every_label_demo.ipynb`. A computer with a valid installation of Python 3 is required. For having this you can run in the anaconda prompt:
* `conda update python` or `conda install python=3.6`
NumPy and Matplotlib packages are used, you can get them with pip tool:
* `pip3 install numpy`
* `pip3 install matplotlib`
The following packages are also required for running the whole notebooks:
- [seaborn](https://seaborn.pydata.org/) for nice visualizations.
- [scikit-learn](http://scikit-learn.org/stable/) ML methods .
- [pandas](https://pandas.pydata.org/) for dealing with the data datasets  
-pyarrow, neccesary for save in .parquet format.
-[joblib](https://pypi.org/project/joblib/) for saving the skcikit-learn grid searches.
-[imageio](https://pypi.org/project/imageio/) and scikit-image for dealing with the images.
You can install them with conda run as below:
* `conda install -c anaconda seaborn`
* `conda install -c anaconda scikit-learn`
* `conda install -c anaconda pandas`
* `conda install -c conda-forge pyarrow` or `conda install -c conda-forge/label/gcc7 pyarrow`
* `conda install -c anaconda joblib`
* `conda install -c conda-forge imageio`
* `conda install -c anaconda scikit-image`

## Running the project

### Obtaining the `.csv` output
The main program is `run.py`, it has to be executed from the scripts folder:
```
/scripts$ python3 run.py
```
It will load the  training data from `train.csv` file, and the test data from
`test.csv`, both included in `../data/`. The program will generate the `.csv`
file that contains the best result in `../out/` directory.

### Testing and plotting

To run different validations, the `hyperparameters_and_plots.py`. It has to be executed from the scripts folder:
```
/scripts$ python3 hyperparameters_and_plots.py
```
The program will output the following progress updates through the standard output:

```
split_ratio=0.85, degree=3, lambda_=0.000000000, Test RMSE=0.771, Train RMSE=0.770, %test=0.79373, %train=0.79570
split_ratio=0.85, degree=3, lambda_=0.000000001, Test RMSE=0.771, Train RMSE=0.770, %test=0.79373, %train=0.79570
split_ratio=0.85, degree=3, lambda_=0.000000002, Test RMSE=0.771, Train RMSE=0.770, %test=0.79373, %train=0.79570
split_ratio=0.85, degree=3, lambda_=0.000000006, Test RMSE=0.771, Train RMSE=0.770, %test=0.79373, %train=0.79570
...
```
So, you will be able to check in the terminal the accuracy (*test%* = accuracy over the test data and *train%* = accuracy over the train data) and the RMSE (over the test data) of the different models.
Finally you will obtain the plots showing the accuracy (over the test data) of different configurations (depending on the hyperparameters `lambda` and `degree`)
