The CustomEncoder is a Python class that provides a scikit-learn-compatible implementation of a custom encoder for categorical variables. This encoder can be used within scikit-learn pipelines for preprocessing data, particularly when dealing with categorical features that require mapping to numerical values.

**Features**
- Implements the BaseEstimator and TransformerMixin classes from scikit-learn for compatibility with scikit-learn pipelines.
- Efficiently learns and applies an encoding mapping to a specific column in input data.
- Generates a unique numerical value for each distinct category, preserving the original order of appearance.
- Handles unseen categories during encoding by assigning a unique value.
- Facilitates seamless integration with scikit-learn workflows and data preprocessing pipelines.

Here's an explanation of each part of the code:

1. The class definition:
   ```python
   class CustomEncoder(BaseEstimator, TransformerMixin):
   ```
   - This line defines the `CustomEncoder` class and specifies that it inherits from `BaseEstimator` and `TransformerMixin`. These are base classes provided by scikit-learn to create custom transformers.

2. The constructor:
   ```python
   def __init__(self, col):
       self.col = col
       self.enc_map = {}
   ```
   - The `__init__` method is called when creating an instance of the `CustomEncoder` class. It takes one argument, `col`, which represents the column name to encode. It initializes two instance variables: `self.col` stores the column name, and `self.enc_map` is an empty dictionary that will hold the encoding mapping.

3. The `fit` method:
   ```python
   def fit(self, X, y=None):
       unique_values = sorted(list(X[self.col].drop_duplicates()))
       self.enc_map = {unique_values[i]: i for i in range(len(unique_values))}
       return self
   ```
   - The `fit` method is responsible for learning the encoding mapping based on the input data `X`. It takes `X` (the input data) and `y` (optional target values) as arguments.
   - The method extracts the unique values from the specified column `self.col` in the input data using `X[self.col].drop_duplicates()`.
   - It then creates a mapping dictionary `self.enc_map` where the unique values are the keys and their corresponding indices are the values. The indices are generated using a range from 0 to the number of unique values.
   - Finally, the method returns `self` to enable method chaining.

4. The `transform` method:
   ```python
   def transform(self, X):
       X_encoded = X.copy()
       X_encoded[self.col] = X_encoded[self.col].replace(list(self.enc_map.keys()),list(self.enc_map.values()))
       return X_encoded
   ```
   - The `transform` method applies the learned encoding mapping to the input data `X`. It takes `X` as an argument.
   - It creates a copy of the input data, `X_encoded`, to avoid modifying the original data.
   - The method replaces the values in the specified column `self.col` of `X_encoded` with their corresponding encoded values using the mapping dictionary `self.enc_map`.
   - Finally, it returns the transformed data `X_encoded`.

In summary, this code defines a custom encoder that can learn and apply an encoding mapping to a specific column in input data. The `fit` method learns the encoding mapping, and the `transform` method applies it to the data. This encoder can be used as part of a scikit-learn pipeline for preprocessing data.
