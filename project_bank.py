import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import classification_report , roc_curve , auc

df1 = pd.read_csv("bank_final.csv")
print(df1.describe())
print(df1.columns)
print(df1.nunique())

print(df1.isna().sum())
df1[df1.duplicated()] #15 duplicates
df1.drop_duplicates(inplace=True) #Duplicate records removed
df1=df1.reset_index(drop=True)

#stripping $ and , sign from currency columns and converting into float64
currency_cols = ['DisbursementGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv','BalanceGross']
for cols in currency_cols:
	df1[cols] = df1[cols].str.replace('$', '').str.replace(',', '')

#OUTPUT VARIABLE ---> MIS_STATUS

print(df1.MIS_Status.value_counts())
#so thid has 2 categorie so this is the best variable for target variable
df1.drop(df1[df1.MIS_Status.isna()].index,inplace=True)
print("mis_null_values:",df1.MIS_Status.isna().sum())

#Null values imputed

#creating output variable
df1['MIS_Status']=df1['MIS_Status'].map({'P I F':0,'CHGOFF':1})

plt.rcParams.update({'figure.figsize':(12,8)})
plt.show()
sns.countplot(x='MIS_Status',data=df1)
plt.title("0 : Non-defaulter and 1: defaulter")
plt.show()
print(df1['MIS_Status'].value_counts())      # approx 40000 defaulters and 110000 non defaulters

len(df1['Name'].unique())
len(df1['City'].unique())
len(df1['State'].unique())
len(df1['Zip'].unique())
len(df1['Bank'].unique())
len(df1['BankState'].unique())
len(df1['CCSC'].unique())
len(df1['ApprovalDate'].unique())
len(df1['ApprovalFY'].unique())
len(df1['Term'].unique())
len(df1['NoEmp'].unique())
len(df1['NewExist'].unique()) #3 categories
len(df1['CreateJob'].unique())
len(df1['RetainedJob'].unique())
len(df1['FranchiseCode'].unique())
len(df1['UrbanRural'].unique())#0-unidentified
len(df1['RevLineCr'].unique()) #wrong entries present
len(df1['LowDoc'].unique()) #wrong entries present
len(df1['ChgOffDate'].unique())
len(df1['DisbursementDate'].unique())
len(df1['DisbursementGross'].unique())
len(df1['BalanceGross'].unique()) #only 3 unique values
len(df1['MIS_Status'].unique()) 
len(df1['ChgOffPrinGr'].unique()) 
len(df1['GrAppv'].unique())
len(df1['SBA_Appv'].unique())

# visualization to check dependency with MIS_status and other columns

####  NewExist  ####
print(df1['NewExist'].value_counts()) #0-->128 wrong entries
df1.drop(df1[df1.NewExist.isna()].index,inplace=True)
sns.countplot(x='NewExist',hue='MIS_Status',data=df1)#Existing business more than new business in dataset
plt.title("MIS_Status Vs NewExist")
plt.show()

print(df1[['NewExist','MIS_Status']].groupby(['NewExist']).mean().sort_values(by='MIS_Status',ascending=False))
#existing business has a little more chance to default than new business
#Imputing with mode
df1.loc[(df1.NewExist !=1) & (df1.NewExist !=2),'NewExist']=1

####  FranchiseCode  ####
print(df1.FranchiseCode.isna().sum())
sns.countplot(x='MIS_Status',hue='FranchiseCode',data=df1)# only few have franchises
plt.title("MIS_Status vs franchisescode")
plt.show()

df1[['FranchiseCode','MIS_Status']].groupby(['FranchiseCode']).mean().sort_values(by='MIS_Status',ascending=True)
#defaulting chances are less for businesses with franchises

####  UrbanRural  ####
df1.drop(df1[df1.UrbanRural.isna()].index,inplace=True)
sns.countplot(x='MIS_Status',hue='UrbanRural',data=df1)#more cases of urban; majority of unidentified is in non-default
plt.title("MIS_Status Vs UrbanRural")
plt.show()
df1[['UrbanRural','MIS_Status']].groupby(['UrbanRural']).mean().sort_values(by='MIS_Status',ascending=False)
#urban business more likely to default

####  RevLineCr  ####
df1['RevLineCr'].value_counts() #0-23659 , T-4819 , (`)-2 , (,)-1
df1.drop(df1[df1['RevLineCr']=='0'].index,inplace=True)
df1.drop(df1[df1['RevLineCr']=='`'].index,inplace=True)
df1.drop(df1[df1['RevLineCr']=='1'].index,inplace=True)
df1.drop(df1[df1['RevLineCr']==','].index,inplace=True)
df1.drop(df1[df1['RevLineCr']=='T'].index,inplace=True)

sns.countplot(x='MIS_Status',hue='RevLineCr',data=df1)#RevLine of credit not availbale for majority of the businesses
plt.title("MIS_Status Vs RevLineCr (0 is urban, 1 is rural)")
plt.show()

df1.drop(df1[df1.RevLineCr.isna()].index,inplace=True)
print(df1.RevLineCr.value_counts())

####  LowDoc  ####
df1['LowDoc'].value_counts() #C-83 , 1-1
df1.drop(df1[df1['LowDoc']=='C'].index,inplace=True)
df1.drop(df1[df1['LowDoc']=='1'].index,inplace=True)
df1['LowDoc']=df1['LowDoc'].map({'Y':0,'N':1})
sns.countplot(x='MIS_Status',hue='LowDoc',data=df1)#majority businesses are not under LowDoc
plt.title('MIS_Status VS LowDoc where y=0 and N=1')
plt.show()

####  ChgOffDate  ####
df1['ChgOffDate_Yes']=0
for i in range(len(df1)):
	try:
		if len(df1['ChgOffDate'][i]):
			df1['ChgOffDate_Yes'][i]=1
	except:
		pass

pd.crosstab(df1.MIS_Status,df1.ChgOffDate_Yes)
plt.show() 
#ChgOffDate present implies it is a defaulter,and if absent, non defaulter with very few exceptions

####  BalanceGross ####
df1['BalanceGross'].value_counts() #only 2 values, rest are 0
df1[['BalanceGross','MIS_Status']].groupby(['BalanceGross']).mean().sort_values(by='MIS_Status',ascending=False)
# dropping balance gross from data

####  ChgOffPrinGr  ####
print(df1.ChgOffPrinGr.isna().sum())
print(df1.GrAppv.isna().sum()) # no null values

pd.crosstab(df1.MIS_Status,df1.ChgOffPrinGr==0)
plt.show()
# if not defaulter then very less chance to have chargeoff amount
# if a defaulter then there are very few cases where the amount is not chargedoff

####  State  ####
df1['State'].value_counts()
df1[['State','MIS_Status']].groupby(['State']).mean().sort_values(by='MIS_Status',ascending=False)
#FL state has highest probabilty to default and VT least
#Imputing the 2 NA values
a=df1.loc[df1.State.isna()]
df1.loc[df1.State.isna(),'City']
df1.loc[df1.State.isna(),'State']#JOHNSTOWN-->NY
df1.loc[df1.Zip==8070,'State']
df1.loc[df1.Name=='SO. JERSEY DANCE/MERRYLEES','State']
#PENNSVILLE-->NJ

sns.countplot(x='State',data=df1) # more loans from CA and NY then FL,TX, OH
plt.show()

df1.loc[df1.City=='PENNSVILLE','State']='NJ'
df1.loc[df1.City=='JOHNSTOWN       NY','State']='NY'

#Replacing the States with their probability values(Mean Encoding)
x=df1[['State','MIS_Status']].groupby(['State']).mean().sort_values(by='MIS_Status',ascending=False)
x['State']=x.index
x=x.set_index(np.arange(0,51,1))
for i in range(len(x)):
    df1=df1.replace(to_replace =x.State[i], value =x.MIS_Status[i]) 
    print(i)

####  City  ####
df1['City'].value_counts()
# sns.countplot(x='MIS_Status',hue='City',data=df1)
# plt.title("City VS MIS_Status")
# plt.show()
df1[['City','MIS_Status']].groupby(['City']).mean().sort_values(by='MIS_Status',ascending=False)

####  BankState  ####
df1['BankState'].value_counts() #Most banks in NC least in PR
df1[['BankState','MIS_Status']].groupby(['BankState']).mean().sort_values(by='MIS_Status',ascending=False)# VA highest, MA least
#Bank in VA state has highest probabilty to default and MA least

####  ApprovalFY  ####
sns.countplot(x='ApprovalFY',data=df1)# more approvals in 1997-1998 and 2004-2007
plt.show()
df1['ApprovalFY'].value_counts() #highest no of approvals in 2006 least in 1962,65,66 
sns.countplot(x='MIS_Status',hue='ApprovalFY',data=df1)
plt.title("MIS_Status Vs ApprovalFY")
plt.show()
df1[['ApprovalFY','MIS_Status']].groupby(['ApprovalFY']).mean().sort_values(by='ApprovalFY',ascending=True)
# if loan is approved before 1982, high probability to default; 1997-2003 very less chance to default 

len(df1.loc[(df1['ApprovalFY']<=1980)])# only 305 approvals before 1980
len(df1.loc[(df1['ApprovalFY']>1980) & (df1['ApprovalFY']<1990)])
len(df1.loc[(df1['ApprovalFY']>1990) & (df1['ApprovalFY']<2003)])
len(df1.loc[(df1['ApprovalFY']>2003)])

df1['ApprovalFY_bin']=pd.cut(df1['ApprovalFY'],bins=[1960,1980,1990,2003,2010],labels=[1,2,3,4])
sns.countplot(x='MIS_Status',hue='ApprovalFY_bin',data=df1)
plt.title("MIS_Status VS ApprovalFY_bin (1960,1980,1990,2003,2010 as 1,2,3,4)")
plt.show()
df1[['MIS_Status','ApprovalFY_bin']].groupby(['ApprovalFY_bin']).mean().sort_values(by='ApprovalFY_bin',ascending=True)

####  ApprovalDate  ####
print(df1.ApprovalDate.isna().sum())
print(len(df1.ApprovalDate.value_counts()))
sns.countplot(x='MIS_Status',hue='ApprovalDate',data=df1)
plt.title("MIS_Status VS ApprovalFY_bin")
plt.show()
df1[['MIS_Status','ApprovalDate']].groupby(['ApprovalDate']).mean().sort_values(by='ApprovalDate',ascending=True)

####  Term  ####
sorted(df1['Term'].unique()) # min=0 , max=480
sns.distplot(df1['Term'])
plt.show()
sns.boxplot(x='MIS_Status',y='Term',data=df1)
plt.title("MIS_Status VS Term")
plt.show()

len(df1.loc[(df1['Term']==0)])
len(df1.loc[(df1['Term']==0) & (df1['MIS_Status']==1)]) #189/202 = 0.935
#If term = 0, almost surely defaults
len(df1.loc[(df1['Term']<=60)])
len(df1.loc[(df1['Term']<=60) & (df1['MIS_Status']==1)])
len(df1.loc[(df1['Term']>120) & (df1['Term']<=180)])
len(df1.loc[(df1['Term']>120) & (df1['Term']<=180)& (df1['MIS_Status']==1)])
len(df1.loc[(df1['Term']>300) & (df1['Term']<=360)])
len(df1.loc[(df1['Term']>300) & (df1['Term']<=360) & (df1['MIS_Status']==1)])
len(df1.loc[(df1['Term']>360)])
len(df1.loc[(df1['Term']>360) & (df1['MIS_Status']==1)])

df1['Term_bin']=0
df1['Term_bin']=pd.cut(df1['Term'],bins=[-1,60,120,180,240,300,360,480],labels=[1,2,3,4,5,6,7])
#cutting the dataframe to 5 year terms ie 60 months each;last bin 10 years

sns.countplot(x='MIS_Status',hue='Term_bin',data=df1)#more defaulters for 0-5 year term; more non defaulters for 5-40 year term
plt.show()
df1[['MIS_Status','Term_bin']].groupby(['Term_bin']).mean().sort_values(by='Term_bin',ascending=True)
#for 0-5 and 30-40 more chance of defaulting; for 5-30 less chance of defaulting
p=pd.DataFrame(df1[['MIS_Status','Term_bin']].groupby(['Term_bin']).mean())
plt.title("Term VS probability to default")
plt.xlabel("Five year Terms")
plt.ylabel("Probability to default")
plt.bar(p.index,p.MIS_Status,color='crimson')
plt.show()

####  NoEmp  ####
sorted(df1['NoEmp'].unique())# min=0 ; max=9999
sns.distplot(df1['NoEmp'])
sns.boxplot(x='MIS_Status',y='NoEmp',data=df1)
plt.title("MIS_Status VS Number of employees")
plt.show()

len(df1.loc[df1['NoEmp']>100]) # only 829 businesses have more than 100 employees
len(df1.loc[df1['NoEmp']<=5]) # 98194 have 5 or less employees
len(df1.loc[df1['NoEmp']<=10]) #122309
len(df1.loc[(df1['NoEmp']>30) & (df1['NoEmp']<=100)])
len(df1.loc[(df1['NoEmp']>100) & (df1['NoEmp']<=10000)])

df1['Emp_bin']=0 # Slicing number of employees into groups
emp_bin=[-1,5,10,15,20,30,100,1000,10000]
emp_lab=list(range(1,9))
df1['Emp_bin']=pd.cut(df1['NoEmp'],bins=emp_bin,labels=emp_lab)

sns.countplot(x='MIS_Status',hue='Emp_bin',data=df1)#both follow same pattern
plt.title("MIS_Status VS Emp_bin")
plt.show()
df1[['MIS_Status','Emp_bin']].groupby(['Emp_bin']).mean().sort_values(by='MIS_Status',ascending=False)
# as the number of employees increase chances of default decrease

####  CreateJob ####
sorted(df1['CreateJob'].unique()) #min=0 ; max=3000
sns.countplot(x='MIS_Status',hue='CreateJob',data=df1)
plt.title("MIS_Status VS CreateJob")
plt.show()

len(df1.loc[df1['CreateJob']>100])# only 44 businesses create more than 100 jobs
len(df1.loc[(df1['CreateJob']>10) & (df1['CreateJob']<=100)])# only 3541 business creates jobs between 10 and 100
len(df1.loc[(df1['CreateJob']>5) & (df1['CreateJob']<=10)])# 4130
len(df1.loc[df1['CreateJob']==0]) # no jobs created for 113064 businesses

df1['CreateJob_bin']=0
df1['CreateJob_bin']=pd.cut(df1['CreateJob'],bins=[-1,0,5,10,100,400,3000],labels=[0,1,2,3,4,5])
sns.countplot(x='MIS_Status',hue='CreateJob_bin',data=df1)#same pattern
plt.title("MIS_Status VS CreateJob_bin where(0,5,10,100,400,3000 is 0,1,2,3,4,5)")
plt.show()
df1[['MIS_Status','CreateJob_bin']].groupby(['CreateJob_bin']).mean().sort_values(by='CreateJob_bin',ascending=True)
#chances of default is least when jobs created is between 10 and 400; highest when >400

####  RetainedJob  ####
sorted(df1['RetainedJob'].unique()) # min=0 ; max=9500
sns.countplot(x='MIS_Status',hue='RetainedJob',data=df1)
plt.title("MIS_Status VS RetainedJob")
plt.show()

len(df1.loc[df1['RetainedJob']>100])# only 194 businesses have retained more than 100 jobs
len(df1.loc[df1['RetainedJob']<10])#135938
len(df1.loc[df1['RetainedJob']==0])#65810
len(df1.loc[(df1['RetainedJob']>100) & (df1['RetainedJob']<=400)])
len(df1.loc[df1['RetainedJob']>400])
len(df1.loc[(df1['RetainedJob']>400) & (df1['MIS_Status']==1)])#no defaulters when Retainedjobs>400

df1['RetainedJob_bin']=0
df1['RetainedJob_bin']=pd.cut(df1['RetainedJob'],bins=[-1,0,5,10,100,400,9500],labels=[0,1,2,3,4,5])
sns.countplot(x='MIS_Status',hue='RetainedJob_bin',data=df1)#if no jobs retained then they generally are biased to be non defaulters
plt.title("MIS_Status VS RetainedJob_bin where 0,5,10,100,400,9500 is 0,1,2,3,4,5")
plt.show()
df1[['MIS_Status','RetainedJob_bin']].groupby(['RetainedJob_bin']).mean().sort_values(by='RetainedJob_bin',ascending=True)
#if retained jobs=0 defaulting very less;then as the jobs increases, the chances of defaulting comes down;defaulters high for 1-10 range

####  DisbursementDate  ####

#Handling the date-time variables 
df1['ApprovalDate']= pd.to_datetime(df1['ApprovalDate']) 
df1['DisbursementDate']= pd.to_datetime(df1['DisbursementDate'])
df1.dtypes
df1['Disbursementyear'] = df1['DisbursementDate'].dt.year
df1['DaysforDibursement'] = df1['DisbursementDate'] - df1['ApprovalDate']
df1['DaysforDibursement'] = df1.apply(lambda row: row.DaysforDibursement.days, axis=1)
df1.drop(df1[df1.DaysforDibursement.isna()].index,inplace=True)
#Removing the Date-time variables ApprovalDate and DisbursementDate
df1=df1.drop(['ApprovalDate','DisbursementDate'],axis=1)

sns.countplot(x='MIS_Status',hue='DaysforDibursement',data=df1)#if no jobs retained then they generally are biased to be non defaulters
plt.title("MIS_Status VS DaysforDibursement")
plt.show()

####  DisbursementGross  ####
print(df1.DisbursementGross.isna().sum())
print(df1.DisbursementGross.value_counts())
df1.DisbursementGross.isna().sum()

####  GrAppv  ####

df1.Bank.isna().sum()# we have 147 nan values in the Bank column so we need to drop these as imputation will not be good for these nominal data
df1.drop(df1[df1.Bank.isna()].index,inplace=True)
df1.Bank.unique()
len(df1.Bank.unique())

####  SBA_Appv  ####
df1.drop(df1[df1.SBA_Appv.isna()].index,inplace=True)
sns.countplot(x='MIS_Status',hue='SBA_Appv',data=df1)
plt.title("MIS_Status VS SBA_Appv")
plt.show()
df1[['MIS_Status','SBA_Appv']].groupby(['SBA_Appv']).mean().sort_values(by='MIS_Status',ascending=False)
#chances of default negligible when SBA_Appv = DisbursementGross; highest when SBA_Appv < DisbursementGross

len(df1.loc[df1.SBA_Appv>df1.GrAppv])#Gross approved amount never less than SBA approved
len(df1.loc[df1.SBA_Appv<df1.GrAppv])
len(df1.loc[df1.SBA_Appv==df1.GrAppv])#8094

#####################################################################

df1['NewExist'].value_counts()
a=df1.loc[(df1.NewExist !=1) & (df1.NewExist !=2)]
x=df1.loc[(df1.NewExist ==1),['NoEmp', 'NewExist', 'CreateJob','RetainedJob']]

df1.loc[(df1.ApprovalFY ==2006),'NewExist'].value_counts()
sns.countplot(df1.CreateJob_bin,hue=df1.NewExist)
plt.title("CreateJob_bin VS NewExist")
plt.show()
df1.loc[(df1.CreateJob_bin ==0),'NewExist'].value_counts()

sns.countplot(df1.NewExist,hue=df1.NoEmp>100)
plt.title("NewExist VS NoEmp>100")
plt.show()
sns.countplot(df1.NewExist,hue=df1.CreateJob>100)
plt.title("NewExist VS CreateJob>100")
plt.show()
#no relation found

df1['LowDoc'].value_counts()
a=df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N')]
print(df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N'),'State'].value_counts())
print(df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N'),'BankState'].value_counts())
print(df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N'),'ApprovalFY'].value_counts())

print(df1.loc[(df1.ApprovalFY==1998),'MIS_Status'].value_counts())
print(df1.loc[(df1.State=='TX'),'MIS_Status'].value_counts())
print(df1.loc[(df1.ApprovalFY==2006),'LowDoc'].value_counts())#when ApprovalFY=2006 ,LowDoc never Y 

a=df1.loc[(df1.LowDoc !='Y') & (df1.LowDoc !='N') & (df1.ApprovalFY!=2006)]
a.State.value_counts()
a.BankState.value_counts()
a.ApprovalFY.value_counts()
a.RevLineCr.value_counts()
#No clear relation for LowDoc with any other feature
#Hence imputation done with mode

df1['RevLineCr'].value_counts()
a=df1.loc[(df1.RevLineCr!='Y')&(df1.RevLineCr!='N')&(df1.RevLineCr!='0')]
df1.loc[(df1.ApprovalFY ==1998),'RevLineCr'].value_counts()
sns.countplot(df1.UrbanRural,hue=df1.RevLineCr)
plt.title("UrbanRural VS RevLineCr") #if urban,more having revline credit;if rural not having revline
plt.show()
sns.countplot(df1.Term_bin,hue=df1.RevLineCr)
plt.title("Term_bin VS RevLineCr")
plt.show()
sns.countplot(df1.LowDoc,hue=df1.RevLineCr)#if under LowDoc, then no revline credit
plt.title("LowDoc Vs RevLineCr")
plt.show()
a[a.LowDoc=='Y']
a=df1.loc[(df1.RevLineCr!='Y')&(df1.RevLineCr!='N')&(df1.RevLineCr!='0')&(df1.RevLineCr!='T')]

#UrbanRural-wrong values
#RevLineCr-wrong values

# changing the data type from object to float
df1.SBA_Appv=df1.SBA_Appv.astype('float')

##############################################################################
####  Feature Selection  ####
df1.columns

features=['UrbanRural','Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',
		'FranchiseCode', 'LowDoc', 'DaysforDibursement','MIS_Status']

df1_clensed=df1[features]
print(df1_clensed)
print(df1_clensed.dtypes)

corri=df1.corr(method='pearson')
corri.style.background_gradient(cmap='coolwarm').set_precision(2)
plt.show()

from sklearn.ensemble import ExtraTreesClassifier

X=df1_clensed.iloc[:,0:10]
Y=df1_clensed.iloc[:,-1]
Y=pd.DataFrame(Y)
Model_ETC=ExtraTreesClassifier(n_estimators=11)
Model_ETC.fit(X,Y)
print(Model_ETC.feature_importances_)
N=12
ind=np.arange(N)

############################## Model building ########################################

from xgboost import XGBClassifier
Model_XGB=XGBClassifier()
Model_XGB.fit(X,Y)

print(Model_XGB.feature_importances_)
plt.bar(range(len(Model_XGB.feature_importances_)),Model_XGB.feature_importances_)
plt.xticks(ind,('State','Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob','FranchiseCode', 'LowDoc', 'DaysforDibursement'),rotation='vertical')
plt.show()

X=df1_clensed.iloc[:,0:10]
Y=df1_clensed.iloc[:,-1]

#Using XGBoost
data_dmatrix = xgb.DMatrix(data=X,label=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
Model_XGB.fit(X_train,Y_train)
print(Model_XGB)
#Train pedictions
X_pred=Model_XGB.predict(X_train)
XGB_predictions_train = [round(value) for value in X_pred]
accuracy_XGB_train = accuracy_score(Y_train, XGB_predictions_train)
print(" TRain Accuracy: %.2f%%" % (accuracy_XGB_train * 100.0))
#Train accuracy=95.76%
#Test predictions
Y_pred=Model_XGB.predict(X_test)
XGB_predictions = [round(value) for value in Y_pred]
# Evaluate predictions
accuracy_XGB = accuracy_score(Y_test, XGB_predictions)
print("Test Accuracy: %.2f%%" % (accuracy_XGB * 100.0))
confusion_matrix(Y_test,XGB_predictions)
matrix = classification_report(Y_test,XGB_predictions)
print(matrix)
#Test accuracy_XGB=94.76%

#ROC of XGB model
fpr_xgb, tpr_xgb, threshold_xgb = roc_curve(Y_test, XGB_predictions)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_xgb, tpr_xgb, 'b', label = 'AUC = %0.2f' % roc_auc_xgb)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#XGBoost with CrossValidation
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1, 
                'max_depth': 5, 'alpha': 10}
CV_rmse = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print(CV_rmse.head())
print((CV_rmse["test-rmse-mean"]).tail(1))
xg_pred = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
#Predictions
dmatrix_xtest=xgb.DMatrix(X_test)
CV_pred=xg_pred.predict(dmatrix_xtest)
CV_predictions=[round(i) for i in CV_pred]
accuracy_CV = accuracy_score(Y_test, CV_predictions)
print("Test Accuracy: %.2f%%" % (accuracy_CV * 100.0))
#XGB_CV=85.14%%
print(Y_test)
print(X_train)

####  Saving model in system  ####
import pickle
pickle.dump(Model_XGB,open('model.pkl','wb'))

####  Load model  ####
Model_XGB=pickle.load(open('model.pkl','rb'))

#########################################################################
