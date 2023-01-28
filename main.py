from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# defining the pop up window
class ABC(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()


# assigning the pop up window for file browsing
root = Tk()
app = ABC(master=root)

# Heading of pop up window
app.master.title("Data Mining Project")

# Frame forBrowsing file
fb = Frame(app)

# label for file
l = Label(fb, text="Filename:")
l.grid(row=0, column=0)

# text field
var = StringVar()
text = Entry(fb, textvariable=var)
text.grid(row=0, column=1)


# defining browse function
def browsefunc():
    filename = filedialog.askopenfilename()
    text.delete(0, END)
    text.insert(0, filename)


# Adding browse button
browsebutton = Button(fb, text="Browse", command=browsefunc)
browsebutton.grid(row=0, column=2)


# defining next function
def donefunc():
    supervisedclas.pack()


# Next button
nextbutton = Button(fb, text="Done", command=donefunc)
nextbutton.grid(row=1, column=2)
fb.pack()

# Frame for Radio button set for survised learning
supervisedclas = Frame(app)

# label for file
labelofsupervisedclas = Label(supervisedclas, text="Select Classification Model")
labelofsupervisedclas.grid(row=0, column=0)

# Setting up Radio Button function
varforsupervisedclas = IntVar()

# Radio Buttons for Linear Regression
KM = Radiobutton(supervisedclas, text="KNeighbour", variable=varforsupervisedclas, value=1)
KM.grid(row=1, column=0)

# Radio button for Decision Trees
DT = Radiobutton(supervisedclas, text="Decision Tree", variable=varforsupervisedclas, value=2)
DT.grid(row=1, column=1)

# Radio button for Naive Bayes
NB = Radiobutton(supervisedclas, text="Naive Bayes", variable=varforsupervisedclas, value=3)
NB.grid(row=1, column=2)

# Radio button for SVM Classifier
SVMC = Radiobutton(supervisedclas, text="SVM", variable=varforsupervisedclas, value=4)
SVMC.grid(row=1, column=3)

# Radio button for Logistic Regression
SVMC = Radiobutton(supervisedclas, text="Logistic Regression", variable=varforsupervisedclas, value=5)
SVMC.grid(row=1, column=4)


# function for supervised classification button click
def supervisedclas_button_function():
    filename = var.get()
    if (varforsupervisedclas.get() == 1):
        result = KNeighbour(filename)
    elif (varforsupervisedclas.get() == 2):
        result = DecisionTreeClas(filename)
    elif (varforsupervisedclas.get() == 3):
        result = NaiveBayesClas(filename)
    elif (varforsupervisedclas.get() == 4):
        result = SVMClas(filename)
    elif (varforsupervisedclas.get() == 5):
        result = Logistic_regression(filename)
    f3.pack()
    textofresult.delete(0, END)
    textofresult.insert(0, result)


# button
supervisedclas_button = Button(supervisedclas, text="Selected", command=supervisedclas_button_function)
supervisedclas_button.grid(row=2, column=0)


# Machine Learning Functions
def Logistic_regression(filename, mylist=[]):
    model = LogisticRegression()
    try:
        data = pd.read_csv(filename)
        d = []
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d) - 1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
        model.fit(X_train, y_train)
        if len(mylist) == 0:
            return model.score(X_test, y_test)
        else:
            res = model.predict(mylist)
            return res
    except:
        print()
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""


def KNeighbour(filename, mylist=[]):
    ans = simpledialog.askinteger("K Nearest Neighbours", "Number of Neighbours",
                                  parent=root,
                                  minvalue=1, maxvalue=1000)
    model = KNeighborsClassifier(n_neighbors=ans)
    try:
        data = pd.read_csv(filename)
        d = []
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d) - 1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
        model.fit(X_train, y_train)
        if len(mylist) == 0:
            return model.score(X_test, y_test)
        else:
            res = model.predict(mylist)
            return res
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""


def KMean(filename, mylist=[]):
    ans = simpledialog.askinteger("K Mean Clustering", "Number of Clusters",
                                  parent=root,
                                  minvalue=1, maxvalue=1000)
    model = KMeans(n_clusters=ans)
    try:
        data = pd.read_csv(filename)
        d = []
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d) - 1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
        model.fit(X_train)
        if len(mylist) == 0:
            return model.score(X_test, y_test)
        else:
            res = model.predict(mylist)
            return res
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""


def DecisionTreeClas(filename, mylist=[]):
    model = DecisionTreeClassifier()
    try:
        data = pd.read_csv(filename)
        d = []
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d) - 1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
        model.fit(X_train, y_train)
        if len(mylist) == 0:
            return model.score(X_test, y_test)
        else:
            res = model.predict(mylist)
            return res
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""


def NaiveBayesClas(filename, mylist=[]):
    model = GaussianNB()
    try:
        data = pd.read_csv(filename)
        d = []
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d) - 1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
        model.fit(X_train, y_train)
        if len(mylist) == 0:
            return model.score(X_test, y_test)
        else:
            res = model.predict(mylist)
            return res
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""


def SVMClas(filename, mylist=[]):
    answer = simpledialog.askstring("SVM", "Kernel Type(linear/poly/rbf/sigmoid)",
                                    parent=app)
    try:
        model = svm.SVC(kernel=answer)
    except:
        messagebox.showinfo("Error", "Wrong kernel is selected")
        return ""
    try:
        data = pd.read_csv(filename)
        d = []
        for i in data.keys():
            d.append(i)
        x = data[d[:len(d) - 1]].values
        y = data[d[-1]].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
        model.fit(X_train, y_train)
        if len(mylist) == 0:
            return model.score(X_test, y_test)
        else:
            res = model.predict(mylist)
            return res
    except:
        messagebox.showinfo("Error", "Wrong data file is selected")
        return ""


# Frame for different models
f3 = Frame(app)

# label for result
l1 = Label(f3, text="Result:")
l1.grid(row=0, column=0)

# text field
vark = StringVar()
textofresult = Entry(f3, textvariable=vark)
textofresult.grid(row=0, column=1)


# way to enter values
def setInputs():
    frameForInputs.pack()


setInputsButton = Button(f3, text="Enter Values To Predict", command=setInputs)
setInputsButton.grid(row=1, column=1)

# frame for inputs
frameForInputs = Frame(app)

# label and input for Pregnancies
l1 = Label(frameForInputs, text="Pregnancies:")
l1.grid(row=0, column=0)

pregEntry = Entry(frameForInputs)
pregEntry.grid(row=0, column=1)

# label and input for Glucose
l2 = Label(frameForInputs, text="Glucose:")
l2.grid(row=1, column=0)

glucoseEntry = Entry(frameForInputs)
glucoseEntry.grid(row=1, column=1)

# label and input for BloodPressure
l3 = Label(frameForInputs, text="BloodPressure:")
l3.grid(row=2, column=0)

BloodPressureEntry = Entry(frameForInputs)
BloodPressureEntry.grid(row=2, column=1)

# label and input for SkinThickness
l4 = Label(frameForInputs, text="SkinThickness:")
l4.grid(row=3, column=0)

SkinThicknessEntry = Entry(frameForInputs)
SkinThicknessEntry.grid(row=3, column=1)

# label and input for Insulin
l5 = Label(frameForInputs, text="Insulin:")
l5.grid(row=4, column=0)

InsulinEntry = Entry(frameForInputs)
InsulinEntry.grid(row=4, column=1)

# label and input for BMI
l6 = Label(frameForInputs, text="BMI:")
l6.grid(row=5, column=0)

BMIEntry = Entry(frameForInputs)
BMIEntry.grid(row=5, column=1)

# label and input for DiabetesPedigreeFunction
l7 = Label(frameForInputs, text="DiabetesPedigreeFunction:")
l7.grid(row=6, column=0)

DiabetesPedigreeFunctionEntry = Entry(frameForInputs)
DiabetesPedigreeFunctionEntry.grid(row=6, column=1)

# label and input for Age
l8 = Label(frameForInputs, text="Age:")
l8.grid(row=7, column=0)

AgeEntry = Entry(frameForInputs)
AgeEntry.grid(row=7, column=1)


def getValuesFromInputs():
    if int(pregEntry.get()) < 0 or int(glucoseEntry.get()) < 0 or int(BloodPressureEntry.get()) < 0 or int(SkinThicknessEntry.get()) < 0 or int(InsulinEntry.get()) < 0 or float(BMIEntry.get()) < 0 or float(DiabetesPedigreeFunctionEntry.get()) < 0 or int(AgeEntry.get())< 0:
        messagebox.showinfo("not valid", "Please enter valid data")
        return


    vlaFromInp = [int(pregEntry.get()), int(glucoseEntry.get())
        , int(BloodPressureEntry.get()), int(SkinThicknessEntry.get()),
                  int(InsulinEntry.get()), float(BMIEntry.get()), float(DiabetesPedigreeFunctionEntry.get()),
                  int(AgeEntry.get())]
    input_data_as_numpy_array = np.asarray(vlaFromInp)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # get selected algo
    filename = var.get()
    result = 0
    if (varforsupervisedclas.get() == 1):
        result = KNeighbour(filename, input_data_reshaped)
    elif (varforsupervisedclas.get() == 2):
        result = DecisionTreeClas(filename, input_data_reshaped)
    elif (varforsupervisedclas.get() == 3):
        result = NaiveBayesClas(filename, input_data_reshaped)
    elif (varforsupervisedclas.get() == 4):
        result = SVMClas(filename, input_data_reshaped)
    elif (varforsupervisedclas.get() == 5):
        result = Logistic_regression(filename, input_data_reshaped)
    if result == 1:
        messagebox.showinfo("Result", "Sick ( {} )".format(result))
    elif result == 0:
        messagebox.showinfo("Result", "Not Sick ( {} )".format(result))

pridictButton = Button(frameForInputs, text="Predict", command=getValuesFromInputs)
pridictButton.grid(row=8, column=1)

# end
app.mainloop()
