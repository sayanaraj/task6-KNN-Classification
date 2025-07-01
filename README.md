# task6-KNN-Classification
# Task 6 – KNN Decision Boundary Visualization

## Objective

The purpose of this task is to visually demonstrate how the **K-Nearest Neighbors (KNN)** algorithm separates different classes using decision boundaries. This is achieved by training a KNN model on the **Iris dataset** using two selected features, and plotting the resulting class regions.

## Dataset

- Name: Iris Dataset  
- Source: [UCI / Kaggle – Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)  
- File Used: `Iris.csv`  
- Target Column: `Species`  
- Features Used:
  - `PetalLengthCm`
  - `PetalWidthCm`

## Tools and Libraries

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

## Implementation Summary

### 1. Data Loading and Preprocessing
- Loaded `Iris.csv` and dropped the `Id` column (if present).
- Encoded `Species` using `LabelEncoder` (Setosa = 0, Versicolor = 1, Virginica = 2).
- Selected only two features: `PetalLengthCm` and `PetalWidthCm`.
- Normalized the features using `StandardScaler`.

### 2. Model Training
- Used `KNeighborsClassifier` with `K = 3`.
- Split the data into 80% training and 20% testing.

### 3. Decision Boundary Plotting
- Generated a mesh grid over the 2D feature space.
- Classified every point on the grid using the trained KNN model.
- Visualized the class boundaries using `contourf()` and actual data points with `scatter()`.

## Output

A visual plot showing:
- Background color regions representing KNN decision boundaries
- Colored dots representing original samples from the Iris dataset
