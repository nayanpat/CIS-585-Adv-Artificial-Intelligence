import csv
import time
from tkinter import INSERT, END, Button

import matplotlib
import numpy
import pandas as pd
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from tkinter import *
import warnings
warnings.filterwarnings('ignore')

'''............................................'''
''' Extracting and cleaning & Organizing data  '''
'''............................................'''

dataTable = pd.read_csv('Movie_Data.csv', encoding='latin-1', index_col=False)

# Drop unnecessary columns
dataTable = dataTable.drop(['View Rating', 'Awards Received', 'Awards Nominated For', 'Boxoffice', 'Netflix Link', 'Summary', 'Image'], axis=1)

# Fill NaN (empty) cells with some text and numeric values
dataTable['Title'] = dataTable['Title'].fillna('Not Available')
dataTable['Genre'] = dataTable['Genre'].fillna('Not Available')
dataTable['Tags'] = dataTable['Tags'].fillna('Not Available')
dataTable['Languages'] = dataTable['Languages'].fillna('Not Available')
dataTable['Director'] = dataTable['Director'].fillna('Not Available')
dataTable['Writer'] = dataTable['Writer'].fillna('Not Available')
dataTable['Actors'] = dataTable['Actors'].fillna('Not Available')
dataTable['IMDb Score'] = dataTable['IMDb Score'].fillna(0.0)
dataTable['IMDb Votes'] = dataTable['IMDb Votes'].fillna(0)

dataTable['IMDb Score'] = pd.to_numeric(dataTable['IMDb Score'])
dataTable['IMDb Votes'] = pd.to_numeric(dataTable['IMDb Votes'])

dataTable.to_csv("cleanedData2.csv", index=False)

'''...................................................'''
'''............Descriptive Data Exploration...........'''
'''...................................................'''
pd.set_option('display.max_columns', None)
print(dataTable.head())
print(dataTable.columns)

# calculating the average of the IMDB scores of the movies in the dataset
scoreAverage = dataTable['IMDb Score'].mean()
print('The overall average of all the IMDB scores of the movies out of 10 is: ', scoreAverage)

# calculating the average of the IMDB votes of the movies in the dataset
voteAverage = dataTable['IMDb Votes'].mean()
print('The overall average of all the IMDB Votes of the movies is: ', voteAverage)

# A histogram showing the frequency distribution of IMDB Score
plt.title("IMDb Scores Distribution")
plt.xlabel('IMDb score')
plt.ylabel('Frequency')
plt.hist(dataTable['IMDb Score'], color="skyblue", edgecolor='black')
plt.show()

plt.figure(figsize=(50,15))
# A histogram showing the frequency distribution of IMDB Votes
plt.title("IMDbbB Votes Distribution")
plt.xlabel('IMDb Votes')
plt.ylabel('Frequency')
plt.hist(dataTable['IMDb Votes'], color="skyblue", edgecolor='black')
plt.show()

# Drawing the bar plot for the genre column
# Grouping the dataframe by "genre" column and then counting each category and displaying first 30 genres with largest count
y = dataTable.groupby("Genre")['Genre'].count().nlargest(30)
x = y.index  # the index(row labels) of the dataframe

plt.figure(figsize=(50, 15))  # setting the plot figure size
ax = sns.barplot(x=y, y=x)  # using the seaborn's barplot method
ax.set(xlabel='Count')
ax.set(ylabel='')
plt.show()

# Filtering out the genres which have a film count greater than 10
genre_filter=dataTable['Genre'].value_counts().loc[lambda x: x>10].to_frame()
print(genre_filter)

# Checking the count of the language of the film
lang = dataTable.groupby('Languages')['Languages'].count().nlargest(15)
y = lang
x = lang.index

plt.figure(figsize=(15, 7))
l = sns.barplot(x=y, y=x)
l.set(title='Distribution of Language of Films', xlabel="Count", ylabel="")
plt.show()

'''...................................................'''
'''...................Modeling........................'''
'''...................................................'''

weight = 40
features = dataTable.drop(columns=['Title'])
target = dataTable['Title']

# Scale numeric values
numTransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# One-hot encode categorical values
catTransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numTransformer, selector(dtype_include='float64')),
        ('cat', catTransformer, selector(dtype_include='category'))])

# Convert your data as numeric values
X = preprocessor.fit_transform(dataTable)
y = np.stack(target.values)

# Create 2 datasets for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Missing step, use `StandardScaler` to normalize numeric values

# Train your model
rangeOfK = range(10)
# algoList = ['kd_tree', 'brute', 'ball_tree', 'auto']
algoList = ['auto']
evalTable = [['' for i in range(4)] for j in range(40)]
print(evalTable)

index = 0
# Evaluating the performance of the model for different values of k and with different types of algorithm
for algo in algoList:
    for rng in rangeOfK:
        movieRecommenderModel = KNeighborsClassifier(n_neighbors=rng+1, algorithm=algo)
        movieRecommenderModel.fit(X_train, y_train)
        yPred = movieRecommenderModel.predict(X_test)
        modelAccuracy = weight * accuracy_score(y_test, yPred)
        modelPrecision = weight * precision_score(y_test, yPred, average='weighted')
        temp1 = str(rng+1)
        temp2 = str(modelAccuracy)
        temp3 = str(modelPrecision)
        evalTable[rng + index][0] = temp1
        evalTable[rng + index][1] = str(algo)
        evalTable[rng + index][2] = temp2
        evalTable[rng + index][3] = temp3
        print('Model Precision for {} algorithm with value of k = {} is {}'.format(algo, rng+1, modelPrecision))
        print('Model Accuracy for {} algorithm with value of k = {} is {}'.format(algo, rng+1,  modelAccuracy))
    index = index + 10
print(evalTable)

# convert array into dataframe and save the dataframe as a csv file
DF = pd.DataFrame(evalTable)
DF.to_csv("result.csv")

# Finally set the best algorithm for testing
movieRecommenderModel = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
movieRecommenderModel.fit(X_train, y_train)

'''...................................................'''
'''............Testing Model With User Input..........'''
'''...................................................'''
#  User Interface Framework
window = Tk()
window.title("Movie Recommender")
window.geometry("500x750")
greeting = Label(text='Welcome to Movie Recommendation App!\n\n', anchor='n')
greeting.config(font =("Courier", 14))
greeting.pack()

# Text Box1 for Movie 1 Entry
inputTxtBox1 = Text(window, height=5, width=20)
inputTxtBox1.insert(INSERT, 'Enter Movie 1 Name')

# Text Box2 for Movie 2 Entry
inputTxtBox2 = Text(window, height=5, width=20)
inputTxtBox2.insert(INSERT, 'Enter Movie 2 Name')

table = Text(window, height=15, width=50)

#Define a function to clear the content of the text widget
def click1(event):
   inputTxtBox1.configure(state=NORMAL)
   inputTxtBox1.delete('1.0', END)
   inputTxtBox1.unbind('<Button-1>', clearText1)

def click2(event):
   inputTxtBox2.configure(state=NORMAL)
   inputTxtBox2.delete('1.0', END)
   inputTxtBox2.unbind('<Button-1>', clearText2)

indices = [[]]
def getRecommendations(X, model):

    global distance, indices

    movieName1 = inputTxtBox1.get('1.0', 'end-1c')
    movieName2 = inputTxtBox2.get('1.0', 'end-1c')

    movieName = [movieName1.strip(), movieName2.strip()]
    print('movieName = ', movieName[0])
    movieId = [0, 0]

    for idx in range(len(movieName)):
        for id, row in dataTable.iterrows():
            if row['Title'].lower() == movieName[idx].lower():
                print('Movie name & id = ', movieName[idx], id)
                movieId[idx] = id
                break

    # # movieId = row_number
    print('Movie ID = ', movieId)
    # Get the row corresponding to the movie of interest
    movie = X[(movieId[0] & movieId[1]), :]
    print('movie = ', movie)

    # Get the indices and distances of the nearest neighbors
    distances, indices = model.kneighbors(movie.reshape(1, -1))

    print('indices', indices)
    # Return the movie titles corresponding to the nearest neighbors
    # return dataTable['Title'].iloc[indices[0]]
    return indices

def displayResults():
    global indices
    table.delete("1.0", END)
    table.tag_config('left', justify='left', font =("Courier", 12))
    table.insert(INSERT, 'Here are the recommended movies: \n\n')
    movieList = dataTable['Title'].iloc[indices[0]].tolist()
    for idx in range(len(movieList)):
        table.insert(INSERT, str(idx+1) + ')   ' + str(movieList[idx]).replace('{', '').replace('}', '') + '\n')
    table.tag_add("left", "1.0", END)
    table.pack()

# Button Creation
resultButton = Button(window, text="Get Results", command=lambda: [getRecommendations(X, movieRecommenderModel), displayResults()])

# Define a function to close the window
def close():
   #win.destroy()
   window.quit()

# Create a Button to call close()
exitButton = Button(window, text= "CLOSE", font=("Calibri", 14, "bold"), command=close)

inputTxtBox1.pack()
inputTxtBox2.pack()
resultButton.pack()
exitButton.pack()

clearText1 = inputTxtBox1.bind('<Button-1>', click1)
clearText2 = inputTxtBox2.bind('<Button-1>', click2)
print('indices', indices)

window.mainloop()
