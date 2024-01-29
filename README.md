# Alphabet Soup Charity Prediction Project
## Neural-Networks


### Project Overview

This GitHub project is dedicated to the development and optimization of a binary classification model for Alphabet Soup. The provided dataset contains information on more than 34,000 organizations that have received funding from Alphabet Soup. The dataset includes various columns capturing metadata about each organization.

### Preprocess the Data
In this step, the dataset is preprocessed using Pandas and scikit-learn's StandardScaler(). The preprocessing steps include:

1. Read in the charity_data.csv to a Pandas DataFrame.
2. Identify the target variable(s) for the model.
3. Identify the feature variable(s) for the model.
4. Drop the EIN and NAME columns.
5. Determine the number of unique values for each column.
6. For columns with more than 10 unique values, determine the number of data points for each unique value.
7. Bin "rare" categorical variables together in a new value, Other.
8. Use pd.get_dummies() to encode categorical variables.
9. Split the preprocessed data into features (X) and target (y).
10. Scale the training and testing features datasets using StandardScaler.
 
### Compile, Train, and Evaluate the Model
In this step, a neural network model is designed, compiled, trained, and evaluated using TensorFlow and Keras. The steps include:

1. Create a neural network model with the appropriate number of input features and nodes for each layer.
2. Create hidden layers with appropriate activation functions.
3. Create an output layer with an appropriate activation function.
4. Compile and train the model.
5. Create a callback to save the model's weights every five epochs.
6. Evaluate the model using test data to determine loss and accuracy.
7. Save and export the results to an HDF5 file named AlphabetSoupCharity.h5.

### Optimize the Model
In this step, the goal is to optimize the model to achieve a target predictive accuracy higher than 75%. Methods for optimization include:

1. Adjusting input data to handle outliers.
2. Modifying the structure of the neural network (adding neurons, layers, or changing activation functions).
3. Tweaking training parameters (epochs, learning rate, etc.).
4. After at least three attempts at optimization, a new Google Colab file named AlphabetSoupCharity_Optimization.ipynb is created. The dataset is preprocessed, and a new neural network model is designed and trained for improved accuracy. The final results are saved and exported to an HDF5 file named AlphabetSoupCharity_Optimization.h5.



