from tkinter import *


window = Tk()
window.title("Movie Recommender")
window.geometry("500x750")
greeting = Label(text='Welcome to Movie Recommendation App!\n\n', anchor='n')
greeting.config(font =("Courier", 14))


# Text Box1 for Movie 1 Entry
inputTxtBox1 = Text(window, height=5, width=20)
inputTxtBox1.insert(INSERT, 'Enter Movie 1 Name')

# Text Box2 for Movie 2 Entry
inputTxtBox2 = Text(window, height=5, width=20)
inputTxtBox1.insert(INSERT, 'Enter Movie 2 Name')

# Button Creation
printButton = Button(window, text="Get Results")

greeting.pack()
inputTxtBox1.pack()
inputTxtBox2.pack()
printButton.pack()

window.mainloop()