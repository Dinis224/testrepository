# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:05:20 2024

@author: gonca
"""
#1.1Linear Algebra, Calculos and Probability
""""EXercise 2c"""
import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**3 - 1/x

def f_prime(x):
    return 3*x**2 + 1/x**2

# Calculate the tangent lines
def tangent_line(x, x0):
    return f_prime(x0) * (x - x0) + f(x0)

# Create x values for plotting
x_vals = np.linspace(0.5, 3, 400) 
"""To plot continuous curves in matplotlib,
 we need to evaluate the function at many 
x-values. The x_vals array is the set of 
x-coordinates at which the function 𝑓(x)) and the tangent lines are calculated, 
allowing for a smooth and detailed plot."""
y_vals = f(x_vals)

# Tangent lines at x = 1 and x = 2
x0_1, x0_2 = 1, 2
tangent_1 = tangent_line(x_vals, x0_1)
tangent_2 = tangent_line(x_vals, x0_2)

# Plotting the function and the tangent lines
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - \frac{1}{x}$', color='blue')
plt.plot(x_vals, tangent_1, label=f'Tangent line at x = {x0_1}', linestyle='--', color='orange')
plt.plot(x_vals, tangent_2, label=f'Tangent line at x = {x0_2}', linestyle='--', color='green')

# Marking the points where the tangents touch the curve
plt.scatter([x0_1, x0_2], [f(x0_1), f(x0_2)], color='red', zorder=5)
plt.text(x0_1, f(x0_1), f'({x0_1}, {f(x0_1):.2f})', fontsize=12, ha='right')
plt.text(x0_2, f(x0_2), f'({x0_2}, {f(x0_2):.2f})', fontsize=12, ha='left')

# Labels and legend
plt.title(r'Plot of $f(x) = x^3 - \frac{1}{x}$ and its Tangent Lines')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.legend()
plt.grid(True)
plt.ylim(-2, 12)

# Show the plot
plt.show()

#%%
"""Exercise 3e"""
"""i and ii"""
import random

def coin_toss(p, N):
    outcomes = []
    for _ in range(N):
        if random.random() < p:
            outcomes.append('heads')
        else:
            outcomes.append('tails')
    return outcomes

def estimate_p(p, N):
    outcomes = coin_toss(p, N)
    nH = outcomes.count('heads')
    nT = outcomes.count('tails')
    estimated_p = nH / N
    return estimated_p

# Set probability of heads
p = 0.6

# Different values of N
N_values = [2, 5, 8, 20, 100, 1000, 10000, 1000000]

# Calculate and print estimated probabilities for each N
estimates = {}
for N in N_values:
    estimate = estimate_p(p, N)
    estimates[N] = estimate
    print(f"N = {N}: Estimated p = {estimate:.4f}")


#%%1.2 Python basics
#1
#a)
def squares_of_evens(lst):
    return[x**2 for x in lst if x % 2 == 0]

#Example usage
input_list = [1,2,3,4,5,6]
result = squares_of_evens(input_list)
print(result)

#%%
#b)
def lengths_of_strings(lst):
    return{s: len(s) for s in lst}

#Example usage
input_list = ["aeiou","python","multivariada","machine learning"]
result = lengths_of_strings(input_list)
print(result)

#%%
#c)
def apply_function_to_list(lst,f):
    return list(map(f, lst))

input_list = [1,2,3,4,5,6]
result = apply_function_to_list(input_list, lambda x: x**2 )
print(result)

#%%
#d)
def squares_of_evens_map_filter(numbers):
    # Filter the even numbers and then map to their squares
    return list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# Example usage:
input_list = [1, 2, 3, 4, 5, 6]
result = squares_of_evens_map_filter(input_list)
print(result)

#%%
#e)
from functools import reduce

def product_of_list(numbers):
    return reduce(lambda x, y: x * y, numbers)

# Example usage:
input_list = [1, 2, 3, 4, 5]
result = product_of_list(input_list)
print(result)  # Output: 120

#%%1.2 Python for Machine Learning
#1
#a) Numpy

def squares_of_evens_numpy(lst):
    return np.array([x**2 for x in lst if x % 2 == 0])

input_list = [1, 2, 3, 4, 5,6]
result = squares_of_evens_numpy(input_list)
print(result)

#%%
#b) Read csv file with Numpy
import numpy as np

def read_csv_to_array(file_path):
    # Load the data from the CSV file into a NumPy array
    data = np.genfromtxt(file_path, delimiter=',', dtype = None, encoding = None, skip_header=1)  # Adjust skip_header if no header exists
    return data

#dtype = None: this tells numpy to automatically infer the data type of the columns
#encoding = None: This allows for reading data without specific encoding
# Example usage:
result = read_csv_to_array('planets.csv')
print(result)

#%%
#b) alternative using pandas
#If you frequently work with mixed data types, consider using the Pandas library, 
#which is specifically designed for handling such cases:

import pandas as pd

def read_csv_to_dataframe(file_path):
    return pd.read_csv(file_path)

# Example usage:
df = read_csv_to_dataframe('planets.csv')
print(df)

#%%
#c) write csv from Numpy
######################################################!!!!!!!!!!!!!!!!!!!!!!!!
import numpy as np

def append_planets_to_csv(file_path, new_data):
    # Append the new data to the existing CSV file
    np.savetxt(file_path, new_data, delimiter=',')

# Example usage:
# New data to append
additional_data = np.array([
    ['Pluto', 39.48, 248.0],  # Example additional planet
    ['Eris', 96.3, 558.0]     # Another example
])

# Append to the existing planets.csv
append_planets_to_csv('planets.csv', additional_data)
print("Additional data appended to planets.csv.")

#%%
#d) Matplotlib

import matplotlib.pyplot as plt

def plot_squares_of_evens(numbers):
    # Filter the even numbers and compute their squares
    evens = [x for x in numbers if x % 2 == 0]
    squares = [x ** 2 for x in evens]
    
    # Plot the squares of even numbers
    plt.plot(evens, squares, marker='o', linestyle='-', color='r')
    plt.title('Squares of Even Numbers')
    plt.xlabel('Even Numbers')
    plt.ylabel('Square of Even Numbers')
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Example usage:
input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plot_squares_of_evens(input_list)

#%%
#e) Matplotlib advanced features

import matplotlib.pyplot as plt

def advanced_plot_squares_of_evens(numbers):
    # Filter the even numbers and compute their squares
    evens = [x for x in numbers if x % 2 == 0]
    squares = [x ** 2 for x in evens]

    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the even numbers (gray dashed line with star markers)
    plt.plot(evens, evens, marker='*', linestyle='--', color='gray', label='Even numbers')
    
    # Plot the squares of even numbers (red dashed line with star markers)
    plt.plot(evens, squares, marker='*', linestyle='--', color='red', label='Squares of even numbers')
    
    # Add title and labels
    plt.title('Even Numbers and Their Squares')
    plt.xlabel('Even Numbers')
    plt.ylabel('Values')
    
    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend()

    # Save the plot to a PDF file
    plt.savefig('plot.pdf')
    
    # Display the plot
    plt.show()

# Example usage:
input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
advanced_plot_squares_of_evens(input_list)

#%%
#f) Pandas

import pandas as pd

def lengths_of_strings_pandas(lst):
    return pd.DataFrame({"string": lst, "length": [len(s) for s in lst]})

# Example usage:
input_strings = ['apple', 'banana', 'cherry', 'date', 'elderberry']
df = lengths_of_strings_pandas(input_strings)
print(df)

#%%
#f) outra forma (mais percetível)

import pandas as pd

def strings_to_length_dataframe(strings):
    # Create a dictionary with strings and their lengths
    data = {'String': strings, 'Length': [len(s) for s in strings]}
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)
    
    return df

# Example usage:
input_strings = ['apple', 'banana', 'cherry', 'date', 'elderberry']
df = strings_to_length_dataframe(input_strings)
print(df)

#%%
#g) Read a csv file with pandas
#If you frequently work with mixed data types, consider using the Pandas library, 
#which is specifically designed for handling such cases:

import pandas as pd

def read_csv_pandas(file_path):
    return pd.read_csv(file_path)

# Example usage:
df = read_csv_pandas('planets2.0.csv')
print(df)

#%%
#h) Read a spreadsheet file with pandas

import pandas as pd
import os.path

def read_file_pandas(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    elif path.endswith(".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError("File format not supported")

#%%
#i) Scikit-learn
#Although scikit-learn is not typically used for basic arithmetic operations like this, 
#using it might make sense in the context of data pipelines where you want to use
# FunctionTransformer or similar transformations as part of preprocessing in a machine learning
# workflow. However, for simpler cases, basic Python and NumPy approaches are more direct.


import numpy as np
from sklearn.preprocessing import FunctionTransformer

def squares_of_evens_with_sklearn(numbers):
    # Define the transformation function to square even numbers
    def square_evens(arr):
        return np.array([x ** 2 for x in arr if x % 2 == 0])
    
    # Create a FunctionTransformer with the custom square function
    transformer = FunctionTransformer(square_evens, validate=False)

    # Transform the input list into a NumPy array
    result = transformer.transform(numbers)
    
    return result

# Example usage:
input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
output_array = squares_of_evens_with_sklearn(input_list)
print(output_array)

#%%
######################################## ver como usar ucimlrepo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#j) working with datasets with pandas
#1)
import pandas as pd

def load_iris_with_pandas():
    # Load the Iris dataset from a URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    
    # Define column names for the dataset
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(url, header=None, names=column_names)
    
    return df

def get_samples_and_features(df):
    # Get number of samples and features
    n_samples = df.shape[0]
    n_features = df.shape[1] - 1  # excluding the species column

    return n_samples, n_features

# Example usage:
iris_df = load_iris_with_pandas()
n_samples, n_features = get_samples_and_features(iris_df)
print(f'Number of samples: {n_samples}')
print(f'Number of features: {n_features}')


#%%
#2)
import pandas as pd

def sepal_length_stats_with_describe(iris_df):
    # Extract the 'sepal length' column
    sepal_length = iris_df['sepal length (cm)']

    # Use the describe() method to get the summary statistics
    stats = sepal_length.describe()

    # Extract mean, standard deviation, and quartiles from the describe() output
    mean = stats['mean']
    std_dev = stats['std']
    quartiles = {
        '25%': stats['25%'],
        '50%': stats['50%'],  # Median (50th percentile)
        '75%': stats['75%']
    }

    return {
        'mean': mean,
        'std_dev': std_dev,
        'quartiles': quartiles
    }

# Example usage:
# Loading Iris dataset with scikit-learn
from sklearn.datasets import load_iris

def load_iris_with_sklearn():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df

# Load the dataset
iris_df = load_iris_with_sklearn()

# Calculate statistics for sepal length
stats = sepal_length_stats_with_describe(iris_df)
print("Mean:", stats['mean'])
print("Standard Deviation:", stats['std_dev'])
print("Quartiles:")
print(stats['quartiles'])

#%%
#3)
import pandas as pd

def filter_sepal_length_above_mean(iris_df):
    # Calculate the mean sepal length
    mean_sepal_length = iris_df['sepal length (cm)'].mean()

    # Filter the rows where sepal length is greater than the mean
    filtered_data = iris_df[iris_df['sepal length (cm)'] > mean_sepal_length]

    return filtered_data

# Example usage:
# Loading Iris dataset with scikit-learn
from sklearn.datasets import load_iris

def load_iris_with_sklearn():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df

# Load the dataset
iris_df = load_iris_with_sklearn()

# Get the data points where sepal length is greater than the mean
filtered_data = filter_sepal_length_above_mean(iris_df)

# Display the filtered data
print(filtered_data)

#%%
import matplotlib.pyplot as plt

# Simple plot
x = [0, 1, 2, 3]
y = [0, 1, 4, 9]

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Plot")
plt.show()


#%%
# Bar chart
categories = ['A', 'B', 'C']
values = [10, 20, 15]

plt.bar(categories, values)
plt.show()














