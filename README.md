<H3>NAME : M.RAJESHKANNAN</H3>
<H3>REGISTER NO: 212221230081</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
### STEP 1:
Importing the libraries<BR>
### STEP 2:
Importing the dataset<BR>
### STEP 3:
Taking care of missing data<BR>
### STEP 4:
Encoding categorical data<BR>
### STEP 5:
Normalizing the data<BR>
### STEP 6:
Splitting the data into test and train<BR>

##  PROGRAM:
```

#import libraries
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

#Read the dataset 
df=pd.read_csv("Churn_Modelling.csv")

#Checking data
df.head()
df.tail()
df.columns

#Check the missing data
df.isnull().sum()

#Check for Duplicates
df.duplicated()

#check for outliers
df.describe()

# dropping string values data from dataset
data = df.drop(['Surname', 'Geography','Gender'], axis=1)

#Checking datasets after dropping string values data from dataset
data.head()

#Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

# Split the dataset
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)

# Training and testing model

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))

```
## OUTPUT:
#### Check the missing data

![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/f1faabdf-0245-4be9-925c-d7d94a546180)

#### Check the duplicates
![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/9fe0dca7-e90c-4ca2-94cc-df30dbd736e9)

#### check for outliers
![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/f4e4801a-9093-4beb-b8cf-2a4c16d92a1e)
![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/dcf3524d-a8a9-4e91-958e-2b294bdcc231)
![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/6f8ff209-c650-41c5-aae3-b2730254bd03)
![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/d8b230ce-41eb-4bb5-8f3d-2880033d2f91)



#### Normailzed datset after few preprocessing
![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/de6284af-0ad4-404b-bb8f-0fc7fcac53da)

#### Training and testing

![image](https://github.com/Rajeshkannan-Muthukumar/Ex-1-NN/assets/93901857/f5a1f92f-d739-4414-8560-dc6787defbad)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


