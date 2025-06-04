# Part 2 - Weather Data Analysis

This assignment involves exploring a real-world weather dataset to understand the relationships between different weather variables and to **predict the minimum daily temperature** using the relevant feature(s).

## Files

- `part2_submission.ipynb` — Jupyter notebook to be completed.
- `Summary of Weather.csv` — The weather dataset for analysis, it contains information on weather conditions recorded on each day at various weather stations around the world. Information includes precipitation, snowfall, temperatures, wind speed and whether the day included thunder storms or other poor weather conditions.

## Objectives

- Explore and analyze the relationship between weather variables.
- Train and evaluate a model to predicts `MinTemp`

## Submission

To complete this part, please include the following item in your repository:

- A **completed Jupyter notebook (.ipynb)** named `part2_submission_YourName`, containing:
  - All code cells completed and executed, with outputs visible
  - All **Short Answer** questions answered

## Marking Criteria

1. **Load and check the dataset**
    - Loaded the dataset in `part2_submission.ipynb`, checked all the variables(features) in dataset
2. **Clean dataset, drop variables**
    - Shown the correlation between variables and `MinTemp`
    - Dropped irrelevant and highly dependent variables
3. **Select the feature(s) for the model and explain**
    - Relevant features are selected with a clear and well-supported explanation
4. **Split Dataset**
    - The dataset is correctly split using given ratio
5. **Model training and testing**
    - Suitable model and algorithm is used for training and testing 
6. **Evaluate and Visualize Model Performance**
    - Evaluate model
    - Plot the predicted vs true values with the regression line
7. **Understand R², RMSE**
    - Clear explanation is provided about what these metrics indicate regarding model performance
8. **Discuss Model Performance**
    - A well-reasoned discussion of the model’s performance based on model performance plot