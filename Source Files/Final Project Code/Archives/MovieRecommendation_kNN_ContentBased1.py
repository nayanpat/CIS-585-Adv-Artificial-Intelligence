import numpy as nm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from tkinter import *
import warnings
warnings.filterwarnings('ignore')

#  User Interface Framework
window = Tk()
window.title("Movie Recommender")
window.geometry("500x750")
greeting = Label(text='Welcome to Movie Recommendation App!\n\n', anchor='n')
greeting.config(font =("Courier", 14))
greeting.pack()

# Not setting index_col=False would set the first column as an index, but we actually need our first column as 'Title'
dataSet = pd.read_csv('Movie_Data.csv', encoding='latin-1', index_col=False)

def clean_text(text):
    if type(text) == str:
        # Convert to lowercase
        text = text.lower()

        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text

textColumns = dataSet.columns[dataSet.dtypes == 'object']

for col in textColumns:
    dataSet[col] = dataSet[col].apply(clean_text)
dataSet.to_csv("cleanedData.csv", index=False)

# Making a pandas dataframe
cleanedData = pd.read_csv('cleanedData.csv')
dataTable = pd.read_table("cleanedData.csv", delimiter=', ')

# Our columns are singular, ie they aren't separated into 'Title' 'Genre' 'Tags' etc, it is all a singular string
dataTable = dataTable[
    'Title,Genre,Tags,Languages,Director,Writer,Actors,View Rating,IMDb Score,Awards Received,Awards Nominated For,Boxoffice,Netflix Link,Summary,IMDb Votes,Image'].str.split(
    ',', expand=True)

# Assign column names to the new dataframe
dataTable.columns = ['Title',
                     'Genre',
                     'Tags',
                     'Languages',
                     'Director',
                     'Writer',
                     'Actors',
                     'View Rating',
                     'IMDb Score',
                     'Awards Received',
                     'Awards Nominated For',
                     'Boxoffice',
                     'Netflix Link',
                     'Summary',
                     'IMDb Votes',
                     'Image']

# Split the data into train and test data.  That can be used later to model prediction and evaluation

# Use TF-IDF vectorization on the first 6 columns to convert text data to numbers
text_data1 = dataTable['Genre'].tolist()
vectorizer1 = TfidfVectorizer()
tfidfMatrix1 = vectorizer1.fit_transform(text_data1)
print(tfidfMatrix1)
denseTfidfMatrix1 = tfidfMatrix1.todense()

print('denseTfidfMatrix1 = ', denseTfidfMatrix1)

text_data2 = dataTable['Tags'].tolist()
vectorizer2 = TfidfVectorizer()
tfidfMatrix2 = vectorizer2.fit_transform(text_data2)
denseTfidfMatrix2 = tfidfMatrix2.todense()

text_data3 = dataTable['Languages'].tolist()
vectorizer3 = TfidfVectorizer()
tfidfMatrix3 = vectorizer3.fit_transform(text_data3)
denseTfidfMatrix3 = tfidfMatrix3.todense()

text_data4 = dataTable['Director'].tolist()
vectorizer4 = TfidfVectorizer()
tfidfMatrix4 = vectorizer4.fit_transform(text_data4)
denseTfidfMatrix4 = tfidfMatrix4.todense()

text_data5 = dataTable['Writer'].tolist()
vectorizer5 = TfidfVectorizer()
tfidfMatrix5 = vectorizer5.fit_transform(text_data5)
denseTfidfMatrix5 = tfidfMatrix5.todense()

text_data6 = dataTable['Actors'].tolist()
vectorizer6 = TfidfVectorizer()
tfidfMatrix6 = vectorizer6.fit_transform(text_data6)
denseTfidfMatrix6 = tfidfMatrix6.todense()

# combine all 6 matrices to a singular matrix
combinedMatrix = hstack((tfidfMatrix1, tfidfMatrix2, tfidfMatrix3, tfidfMatrix4, tfidfMatrix5, tfidfMatrix6))
print('combinedMatrix = \n', combinedMatrix)

# applying KNN
movieRecommenderModel = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=5)
movieRecommenderModel.fit(combinedMatrix)

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
    movieId = [0, 0]

    for idx in range(len(movieName)):
        for id, row in dataTable.iterrows():
            if row['Title'] == movieName[idx].lower():
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
resultButton = Button(window, text="Get Results", command=lambda: [getRecommendations(combinedMatrix, movieRecommenderModel), displayResults()])

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
