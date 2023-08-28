
###     Importing All Libraries Needed For Buiding The Model 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


###  Creating Dataframe From Dataset -  " FRAMINGHAM CARDIOVASCULAR DISEASE DATASET "

df = pd.read_csv(r'C:\Users\Vindo Singh\Desktop\Revision\framingham.csv')
'''
print(df)
'''

###  Exploratory Data Anaysis (EDA)
'''
print(df.info())
print(df.size)
print(df.shape)
print(df.describe())
print(df.dtypes)
print(df['education'].describe())
print(df.isnull().sum())
'''

df.dropna(subset=['education'], inplace=True)
df.dropna(subset=['cigsPerDay'], inplace=True)
df.dropna(subset=['BPMeds'], inplace=True)
df.dropna(subset=['totChol'], inplace=True)
df.dropna(subset=['BMI'], inplace=True)
df.dropna(subset=['heartRate'], inplace=True)
df.dropna(subset=['glucose'], inplace=True)

'''
print(df)
print(df.isnull().sum())
'''

###  Visualization
'''
# Histogram for age
plt.hist(df['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()
'''
'''
# Barchart for Gender
x = ["Females","Males"]
m = w = 0
for i in df['male']:
    if i==0:
        w+=1
    else:
        m+=1
y = [w,m]

plt.bar(x,y,color=['lime','blue'])
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Gender Distribution")
plt.show()
'''
'''
# Pie Chart for Education Distribution
labels = df['education'].unique()
size = df.groupby('education',sort=False).size()
plt.pie(size,labels=labels,autopct='%1.1f%%')
plt.title("Education Distribution")
plt.show()
'''
'''
# Barchart for Smoking Status
x = ["Non-Smokers","Smokers"]
ns = s = 0
for i in df['currentSmoker']:
    if i==0:
        ns+=1
    else:
        s+=1
y = [ns,s]

plt.bar(x,y,color=['lime','red'])
plt.xlabel("Smoking Status")
plt.ylabel("Count")
plt.title("Count of Smokers and Non-smokers")
plt.show()
'''
'''
# Histogram for Cigrettes per day
plt.hist(df['cigsPerDay'],bins = [0,5,10,15,20,25,30,35,40,45])
plt.xlabel('Cigrettes Per Day')
plt.ylabel('Frequency')
plt.title('Distribution of CigsPerDay')
plt.show()
'''
'''
# Barchart for Blood Pressure Medition
x = ["Patients Not On BP Meds","Patients On BP Meds"]
nbm = bm = 0
for i in df['BPMeds']:
    if i==0:
        nbm+=1
    else:
        bm+=1
y = [nbm,bm]

plt.bar(x,y,color=['lime','red'])
plt.xlabel("BP Medition Status")
plt.ylabel("Count")
plt.title("Count of Patients On or Not On Blood Pressure Medition")
plt.show()
'''
'''
# Barchart for Patients Having Prevalent Stroke
x = ["Not Prevalent Stroke","Have Prevalent Stroke"]
nps = ps = 0
for i in df['prevalentStroke']:
    if i==0:
        nps+=1
    else:
        ps+=1
y = [nps,ps]

plt.bar(x,y,color=['lime','red'])
plt.xlabel("Prevalent Stroke Status")
plt.ylabel("Count")
plt.title("Count of Patients Having or Not Having Prevalent Stroke")
plt.show()
'''
'''
# Barchart for Patients Having Daibetes
x = ["Non-Diabetic","Diabetic"]
nd = d = 0
for i in df['diabetes']:
    if i==0:
        nd+=1
    else:
        d+=1
y = [nd,d]

plt.bar(x,y,color=['lime','red'])
plt.xlabel("Diabetes Status")
plt.ylabel("Count")
plt.title("Count of Patients Having or Not Having Diabetes")
plt.show()
'''

'''
# Barchart for Patients Having Prevalent Hypertension
x = ["Not Prevalent Hypertension","Have Prevalent Hypertension"]
nph = ph = 0
for i in df['prevalentHyp']:
    if i==0:
        nph+=1
    else:
        ph+=1
y = [nph,ph]

plt.bar(x,y,color=['lime','red'])
plt.xlabel("Prevalent Hypertension Status")
plt.ylabel("Count")
plt.title("Count of Patients Having or Not Having Prevalent Hypertension")
plt.show()
'''
'''
# Line Plot to Show Mean Cholesterol Level by Age
mean_cholesterol = df.groupby('age')['totChol'].mean()

plt.plot(mean_cholesterol.index, mean_cholesterol.values,color="red")
plt.xlabel('Age')
plt.ylabel('Mean Cholesterol Level')
plt.title('Mean Cholesterol Level by Age')
plt.show()
'''
'''
# Line Plot to Show Relationship Between Age And Systolic Blood Pressure
mean_SysBP = df.groupby('age')['sysBP'].mean()

plt.plot(mean_SysBP.index, mean_SysBP.values,color="red")
plt.xlabel('Age')
plt.ylabel('Mean Systolic Blood Pressure')
plt.title('Mean Systolic Blood Pressure by Age')
plt.show()

# Line Plot to Show Relationship Between Age And Diastolic Blood Pressure
mean_diaBP = df.groupby('age')['diaBP'].mean()

plt.plot(mean_diaBP.index, mean_diaBP.values,color="red")
plt.xlabel('Age')
plt.ylabel('Mean Diastolic Blood Pressure')
plt.title('Mean Diastolic Blood Pressure by Age')
plt.show()

# Line Plot to Show Relationship Between Age And Heart Rate
mean_heartRate = df.groupby('age')['heartRate'].mean()

plt.plot(mean_heartRate.index, mean_heartRate.values,color="red")
plt.xlabel('Age')
plt.ylabel('Mean Heart Rate')
plt.title('Mean Heart Rate by Age')
plt.show()

# Line Plot to Show Relationship Between Age And Body Mass Index
mean_BMI = df.groupby('age')['BMI'].mean()

plt.plot(mean_BMI.index, mean_BMI.values,color="red")
plt.xlabel('Age')
plt.ylabel('Mean Body Mass Index')
plt.title('Mean Body Mass Index by Age')
plt.show()

# Line Plot to Show Relationship Between Age And Glucose Level
mean_glucose = df.groupby('age')['glucose'].mean()

plt.plot(mean_glucose.index, mean_glucose.values,color="red")
plt.xlabel('Age')
plt.ylabel('Mean Glucose Level')
plt.title('Mean Glucose Level by Age')
plt.show()
'''
'''
# Bar Graph for Patients having 10 year Risk of Coronary Heart Disease (CHD)
x = ['Having CHD','Not Having CHD']
nchd = chd = 0
for i in df['TenYearCHD']:
    if i == 1:
        chd += 1
    else:
        nchd += 1
y = [chd,nchd]

plt.bar(x,y,color=["red","lime"])
plt.xlabel("CHD Status")
plt.ylabel("frequency")
plt.title("Patients having 10 year Risk of Coronary Heart Disease")
plt.show()
'''

# Input and output

# Input  =  male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose.
# Output =  TenYearCHD
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
'''
print(x,"\n",y)
'''

# Train and Test Variables
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state = 0)
'''
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

'''


# As Dataset Has only 3% data of  CHD Patient so it can make model Lean towards Majority
# So, using oversampling to overcome this
# Apply SMOTE oversampling
smote = SMOTE()
x_train_oversampled, y_train_oversampled = smote.fit_resample(x_train, y_train)



# Run Regressor
model = LogisticRegression(max_iter=10000)


# Fit Model
model.fit(x_train_oversampled, y_train_oversampled)

# Predict the Output
y_pred = model.predict(x_test)

# Evaluation of Model


acc =  accuracy_score(y_test,y_pred) * 100   # For Accuracy
print("Accuracy of Model is :: ",format(acc,'.2f'),"%")


report = classification_report(y_test, y_pred)  # Calculate precision, recall, F1-score, and support
print(report)

auc_roc = roc_auc_score(y_test, y_pred)
print("AUC-ROC Score: {:.2f}".format(auc_roc))


'''
pred = model.predict([[1, 55, 2, 1, 20, 1, 0, 1, 0, 250, 140, 90, 30, 70, 120]])
print(pred)
'''


# Main Predictor Model Program
'''
print("\n")
print("*"*83)
print("\t\t\t  Heart Disease Prediction Model  ")
print("*"*83)
print("\n")

print("-"*30)
print("  Patient Details")
print("-"*30)
print("\n")

print("Gender - input 0 for 'female' and 1 for 'male'")
gen = int(input("Gender ::  "))
print("\n")

print("Age - It should be 'integer' not 'fraction'")
age = int(input("Age  ::  "))
print("\n")

print("Education - 1, 2, 3 & 4  for HighSchool, Intermediate, UG, Advanced Degree Completion respectively")
edu = int(input("Educational Status :: "))
print("\n")

print("Current Smoker - 1 for 'smoker' and 0 for 'non-smoker'")
smkr = int(input("Current Smoker  :: "))
print("\n")

print("Cigrettes Per Day - Only Integers, 0 for 'non-smoker'")
cigsPerDay = int(input("Cigrettes Per Day :: "))
print("\n")

print("BP Medition Status - 1 for 'Yes' and 0 for 'No'")
bpMeds = int(input("BP Medition Status :: "))
print("\n")

print("Prevalent Stroke Status - 1 for 'Yes' and 0 for 'No'")
stroke = int(input("Prevalent Stroke Status :: "))
print("\n")

print("Prevalent Hypertension Status - 1 for 'Yes' and 0 for 'No'")
hyp = int(input("Prevalent Hypertension Status :: "))
print("\n")

print("Diabetic Status - 1 for 'Yes' and 0 for 'No'")
diabetes = int(input("Diabetic Status :: "))
print("\n")

print("Total Cholestrol Level - in milligrams per deciliter (mg/dL)")
chol = float(input("Total Cholestrol Level :: "))
print("\n")

print("Systolic Blood Pressure -  in millimeters of mercury (mmHg)")
sysbp = float(input("Systolic Blood Pressure :: "))
print("\n")

print("Diastolic Blood Pressure - in  millimeters of mercury (mmHg)")
diabp = float(input("Diastolic Blood Pressure :: "))
print("\n")

print("Body Mass Index - Weight-to-height ratio for body fatness")
bmi = float(input("Body Mass Index :: "))
print("\n")

print("Heart Rate - Heart Beats per minute")
heart_rate = int(input("Heart Rate :: "))
print("\n")

print("Glucose Level - milligrams per deciliter (mg/dL)")
gulcose = int(input("Glucose Level :: "))
print("\n")

Input = [[gen, age, edu, smkr, cigsPerDay, bpMeds, stroke, hyp, diabetes, chol, sysbp, diabp, bmi, heart_rate, gulcose]]
Prediction_by_model = model.predict(Input)

if Prediction_by_model[0] == 1:
    print("*"*56)
    print("According To Model ")
    print("Based On Given Information About Patient")
    print("Model Predicted That Patient Have Coronary Heart Disease")
    print("*"*56)

elif Prediction_by_model[0] == 0:
    print("*"*56)
    print("According To Model ")
    print("Based On Given Information About Patient")
    print("Model Predicted That Patient Not Have Coronary Heart Disease")
    print("*"*56)

'''    






