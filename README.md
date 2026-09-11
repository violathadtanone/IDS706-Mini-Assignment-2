# IDS 706 Mini Assignment 2: Start Your First Data Analysis - 10 Sep 2026

## Project Description
This is the 2nd mini assignment under IDS 706 with the purpose for data analysis. The first part will cover the usage of pandas and polars with common data manipulation and visualisation. The latter part of this assignment will cover experimentation with Rust on Jupyter notebook.


## Project Structure 
```bash
IDS706-Mini-Assignment-2
├── .gitignore
├── requirements.txt         # List of packages required for installation
├── analysis_query.ipynb     # Jupyter Notebook for Data Analysis
├── rust_exploration.ipynb   # Jupyter Notebook for Rust Exploration
├── Images/                  # Images used for supporting README.md explantion
└── README.md                # Project documentation
```

## Dataset Description 
The dataset uses for this assignment is Sleep_health_and_lifestyle_dataset.csv from Kaggle. It covers a wide range of variables related to sleep and daily habits such as gender, age, occupation, sleep duration, quality of sleep, and the presence or absence of sleep disorders. This can be download from: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset


## Overall Setup Instructions
### 1. Creat GitHub repository
General:
- Name the repository with `IDS706-Mini-Assignment-2`.

Configuration:
- Add README - Toggle On option.
- Add .gitignore - Select Python.
- Proceed to create repository.
<br><br>

### 2. Clone repository in VS Code
- Open Command Palette and select `Git: Clone`.
- Paste GitHub repository URL (e.g. https://github.com/violathadtanone/IDS706-Mini-Assignment-2).
- Select the local folder to continue the development.
<br><br>

### 3. Set up a Python virtual environment
- Create and activate the virtual environment on Terminal with the code below:
```bash
python -m venv .venv
source .venv/bin/activate
```
- Upgrade pip to ensure that it is compatible with the current Python version before installing other packages. We also include upgrade `ipykernel` since it is key for the assignment.
```bash
python -m pip install --upgrade pip
python -m pip install --upgrade ipykernel
```
<br><br>

### 4. Create requirement file for project dependencies (e.g. python packages required)
- Create a new file called `requirements.text` in the project root and add packages below in the file.
```
pandas
polars
scikit-learn
matplotlib
```
- Install the requirements in the visual environment `(.venv) (base)` with the code below in Terminal:
```bash
python -m pip install -r requirements.txt
```
<br><br>

## Setting up Pandas and Polars
### 1. Create new Jupyter Notebook file
- Create a new file called `analysis_query.ipynb`
<br><br>

### 2. Import required library
- In `analysis_query.ipynb`, use the code below to import the installed packages during setup.
```python
import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
```
- Check whether the packages for pandas, polars and numpy have been properly imported into the enviroment. This should return the version number.
 ```python
print(pd.__version__)
print(pl.__version__)
print(np.__version__)
```
<br><br>

### 3. Import dataset
- Add the dataset `Sleep_health_and_lifestyle_dataset.csv` into the project root and import the dataset into `analysis_query.ipynb` using the code below:
```python
sleep_data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
```
<br><br>

## Setting up Rust
### 1. Install Rust and validate
- Use the code below in Terminal. This should return the version number for sucessful installation.
 ```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version
cargo --version
```
<br><br>

### 2. Rust Jupyter kernel
- Upon successful installation in terminal, this should return as "Installation complete".
 ```bash
cargo install evcxr_jupyter
evcxr_jupyter --install
```
<br><br>

### 3. Create new Jupyter Notebook file for Rust
- Create a new file called `rust_analysis_query.ipynb`.
- Select `Rust` as the kernel. If this does not appear, save the existing works and `⌘ + SHIFT + P` then choose `> Developer: Reload Window`.
<br><br>

## Key Highlights from Data Analysis Results
- More details of analysis conducted can be found in [].

### Gender vs Sleep Duration

- In our small samples of 374 observations, we found that female individuals on average slept slightly longer than male, which aligned with the existing research (see https://pmc.ncbi.nlm.nih.gov/articles/PMC4164903/).
- It appeared that female from our dataset slept around 7.23 hours per day, while it was around 7.04 hours per day for male. 
- However, because the dataset does not provide sufficient information on the participants’ health status, it is difficult to determine whether the observed difference in sleep duration is associated with gender or other underlying characteristics.

![Sleep Duration by Gender](Image/gender_vs_sleep.png)

### Sleep Duration in Female
- To further explore this pattern, we created a subset containing only 185 female samples to examine whether the relationships observed in the overall dataset persisted within this group.
- For Age, the dataset showed a clear trend that female, who was betwebetween 51 - 60 years old tend to sleep longer.

- For Daily Step Level, it has been initially hypothesized that the more steps female take per day, the longer her daily sleep duration. Nevertheless, the barchart diagram came back with surprising results, where female with lower daily steps tend to have longer duration of sleep at 7.84 hours. Hence we proceeded to perform multiple regression models to understand further of such perplex relationship.

<p align="center">
  <img src="Image/female_vs_age.png" width="45%">
  <img src="Image/female_vs_step.png" width="45%">
</p>

### Machine Learning Results
- Machine learning was used to explore a relationship between `'Daily Steps'` and `'Sleep Duration'`, where tree-based models demonstrated good performance as compared to basic linear regression.
- The model was later improved by adding `'Age'` variable and incorporating Gradient Boosting Model, which works well with non-linear model. 
- The final results showed that all three models perform well, with Random Forest performing best (R² = 0.919, RMSE = 0.225), indicating slightly higher predictive accuracy than Gradient Boosting and Decision Tree.
- Hence adding Age dramatically improves the model performance, with R² increasing from 0.74 to 0.91 for Decision Tree and RMSE decreasing from 0.45 to 0.23. This suggests Age is a highly informative predictor of the target variable.

![Machine Learning Scenario 2](Image/ML_scenario2.png)

## Pandas vs Polars Performance
- Polars was faster than Pandas for the data analysis section (1.00s vs. 1.21s), but slower for the machine learning section (1.69s vs. 1.35s). 
- Overall, this partially aligns with the general consensus that Polars can outperform Pandas, particularly for data manipulation
- However, the results also show that Polars’ performance depends on the type of task, and it may not be faster when using tools like scikit-learn.

![Pandas vs Polars](Image/pandas_vs_polars.png)