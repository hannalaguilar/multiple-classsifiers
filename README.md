# Ensemble multiple classifiers (MAI-SEL)

The goal of this project is to implement the Random Forest and Decision Forest 
algorithms, which are based on the Decision Tree algorithm.

**The final report is in the docs folder.**

## How tu run the code

### Create an environment and activate it:
 ```
conda create --name sel-py310 python=3.10
conda activate sel-py310
pip install -r requirements.txt
 ```

### Run main.py:
 ```
python -m src.main
 ```

Example of output:

```


You will train and test the following datasets: ['titanic', 'iris', 'glass', 'wine-red', 'wine-white']
---------------------------------------TITANIC---------------------------------------
{'algorithm': random forest, 'n_trees': 1, 'F': 1, 'train_acc': 0.756, 'test_acc': 0.68, 'feature_importance': [('Parch', 0.421), ('Age', 0.195), ('Pclass', 0.17), ('Embarked', 0.079), ('Fare', 0.067), ('Sex', 0.058), ('SibSp', 0.011)]}
{'algorithm': random forest, 'n_trees': 1, 'F': 2, 'train_acc': 0.79, 'test_acc': 0.71, 'feature_importance': [('Embarked', 0.34), ('Fare', 0.233), ('Sex', 0.191), ('Age', 0.156), ('SibSp', 0.032), ('Parch', 0.029), ('Pclass', 0.019)]}
{'algorithm': random forest, 'n_trees': 1, 'F': FMethodRF.LOG, 'train_acc': 0.805, 'test_acc': 0.774, 'feature_importance': [('Parch', 0.361), ('Sex', 0.222), ('Fare', 0.132), ('Age', 0.108), ('Embarked', 0.092), ('Pclass', 0.08), ('SibSp', 0.005)]}
{'algorithm': random forest, 'n_trees': 1, 'F': FMethodRF.SQRT, 'train_acc': 0.859, 'test_acc': 0.823, 'feature_importance': [('Sex', 0.333), ('SibSp', 0.209), ('Fare', 0.126), ('Age', 0.101), ('Parch', 0.096), ('Pclass', 0.077), ('Embarked', 0.057)]}

...

```

### Generate figures
 ```
python src/analysis.py 
 ```

The figures are saved in the folder docs/figures.

### Run tests.py:
 ```
pytest -xs tests/tests.py --count 5
 ```
As the data is randomly generated, tests can be run multiple times to
achieve better test coverage.

Example of output:

 ```
.accuracy sklearn:0.921, accuracy my algorithm: 0.921
.accuracy sklearn: 0.453, accuracy my algorithm: 0.413
.accuracy sklearn: 0.653, accuracy my algorithm: 0.640
 ```
