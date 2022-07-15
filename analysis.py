from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy.stats
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
import math
from matplotlib.ticker import MaxNLocator

# code example for one participant

data=pd.read_csv("/home2/s3884457/FullDataset_6h_2204.csv")

p117129 = data.loc[data["ParticipantNumber"]==117129]
p117129=p117129[p117129.questionListName != 'Morning assessment 1']
p117129=p117129[p117129.questionListName != 'Morning Assessment']

# drop empty columns

p117129=p117129.dropna(axis=1, how='all')
p117129.shape

# drop columns with all 0s

columns0=[]

for i in range(len(p117129.columns)):
    if p117129.iloc[:, i].sum() ==0:
        columns0.append(p117129.columns[i])

p117129.drop(columns0, axis=1, inplace=True)
p117129.shape

# drop rows with no passive data 

p117129 = p117129.drop(p117129.index[253:])
p117129.shape

# loading data after more processing within caret package

scaled=pd.read_csv("/home2/s3884457/p117129_scaled2704.csv")


date = p117129['Date'].astype(str).str[0:10]
#date.value_counts(sort=False)


passive= scaled.iloc[:,5:]
na_sum=scaled['na_sum']


# time series splitting

def tssplit(data, days_train, days_test):
    train_len = round(date.value_counts().mean()*days_train)
    test_len = round(date.value_counts().mean()*days_test)
    n_splits = len(data)-train_len-test_len
    trainTestSplit = []
    for i in range(n_splits):
        it=[list(range(i,i+train_len)),list(range((i+train_len),(i+train_len+test_len)))]
        trainTestSplit.append(it)
    return trainTestSplit


# using one day of data for training, one day for testing

trainTestSplit = tssplit(passive, 1, 1)

# random forests

it=0
#y_pred_train=pd.DataFrame()
y_pred_test=pd.DataFrame()
rfimp=pd.DataFrame()


for trainCvIndices, testIndices in trainTestSplit:
    # splitting train and test sets
    XTrainCv, yTrainCv = passive.iloc[trainCvIndices, :], na_sum.iloc[trainCvIndices,]
    XTest, yTest  = passive.iloc[testIndices,:]   , na_sum.iloc[testIndices,]
    
    lr=LinearRegression()
    imp=IterativeImputer(estimator=lr, tol=1e-10,max_iter=30,verbose=2,imputation_order='roman')
    model=imp.fit(XTrainCv)
    XTrainCv=pd.DataFrame(model.transform(XTrainCv))  # impute train and test sets seperately
    XTest=pd.DataFrame(model.transform(XTest))  # using the same impputation model for test set 
    XTrainCv.columns = list(passive.columns)
    XTest.columns = list(passive.columns)

    # cv for inner loop 
    trainTestSplit_inner = LeaveOneOut()
        
        
    estimator = RandomForestRegressor()
    
    # param space 
    
    param_grid = { 
        "n_estimators" : [100, 300, 500],
        "max_depth": [5, 10, 50],
        "min_samples_split": [2, 5,10,20],
        "min_samples_leaf": [2, 5,10,20],
        "max_features"      : ["sqrt"]
        
    }

    # grid search
    grid = GridSearchCV(estimator, param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=trainTestSplit_inner)
    
    grid.fit(XTrainCv, yTrainCv)
         
    it+=1
    # predictions on test set made by model with best hyperparameters
    test = pd.DataFrame( { "actual":  yTest, "predicted": grid.predict(XTest), 'it': it } )  
    y_pred_test=y_pred_test.append(test)    
    
    importances = grid.best_estimator_.feature_importances_  # feature importance 
    feature_list = list(passive.columns)
    feature_importance= sorted(zip(importances, feature_list), reverse=True)
    df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])
    rfimp=rfimp.append(df) 
    
y_pred_test.to_csv("/home2/s3884457/p117129_na_rf12_2605.csv", sep=',', na_rep='NA')
rfimp.to_csv("/home2/s3884457/p117129_na_rfimp_2605.csv", sep=',', na_rep='NA')


# elastic net

it=0
#y_pred_train=pd.DataFrame()
y_pred_test=pd.DataFrame()
co=pd.DataFrame()


for trainCvIndices, testIndices in trainTestSplit:
    # splitting train and test sets
    XTrainCv, yTrainCv = passive.iloc[trainCvIndices, :], na_sum.iloc[trainCvIndices,]
    XTest, yTest  = passive.iloc[testIndices, :]   , na_sum.iloc[testIndices,]
    
    lr=LinearRegression()
    imp=IterativeImputer(estimator=lr, tol=1e-10,max_iter=30,verbose=2,imputation_order='roman')
    model=imp.fit(XTrainCv)
    XTrainCv=pd.DataFrame(model.transform(XTrainCv))  # impute train and test sets seperately
    XTest=pd.DataFrame(model.transform(XTest))  # using the same impputation model for test set 
    XTrainCv.columns = list(passive.columns)
    XTest.columns = list(passive.columns)
    
    
    # cv for inner loop 
    trainTestSplit_inner = LeaveOneOut()
        
        
    model = ElasticNet()

    # param space
    grid = dict()
    grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    grid['l1_ratio'] = arange(0, 1, 0.01)

    # grid search
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=trainTestSplit_inner, n_jobs=-1)
    
    results = search.fit(XTrainCv, yTrainCv)
    #best=grid_results.best_estimator_
    
    # predictions on test set made by model with best hyperparameters
    it+=1
    test = pd.DataFrame( { "actual":  yTest, "predicted": results.predict(XTest), 'it': it } )  
    y_pred_test=y_pred_test.append(test)    
    
    c=results.best_estimator_.coef_  # coefficients as feature importance scores
    feature_list = list(passive.columns)
    feature_importance= sorted(zip(c, feature_list), reverse=True)
    df = pd.DataFrame(feature_importance, columns=['coef', 'feature'])
    coef_df=pd.DataFrame({"col":passive.columns, "coef":c})
    co=co.append(coef_df) 
    
    
y_pred_test.to_csv("/home2/s3884457/p117129_na_en12_2605.csv", sep=',', na_rep='NA')    
co.to_csv("/home2/s3884457/p117129_na_enimp_2605.csv", sep=',', na_rep='NA')


# results

# calculating r squared, MAE, RMSE by distance from training sets for each participant 
# example for one person
y_pred_test=pd.read_csv("~/Downloads/p117113_na_rf12_2505.csv")

df= pd.DataFrame()
#y_pred_test['idx'] = y_pred_test.index
for i in y_pred_test['it'].astype(int).unique():
    a=y_pred_test.loc[y_pred_test["it"]==i].reset_index(level=0)
    a.index += 1 
    df=df.append(a)
    
df['distance'] = df.index
df=df.loc[:, df.columns != 'index']    

df=df.loc[df['distance']<=df['distance'].max()/2]

distance_df113=pd.DataFrame()

idx=0
for i in df['distance'].astype(int).unique():
    r=df.loc[df["distance"]==i]['predicted'].corr(df.loc[df["distance"]==i]['actual'])
    r2=r**2
    mae=metrics.mean_absolute_error(df.loc[df["distance"]==i]['predicted'], df.loc[df["distance"]==i]['actual'])
    rmse=metrics.mean_squared_error(df.loc[df["distance"]==i]['predicted'], df.loc[df["distance"]==i]['actual'], squared=False)
    d= pd.DataFrame({'distance':i, 'R-Squared':r2, 'MAE':mae, 'RMSE':rmse}, index=[idx])
    distance_df113=distance_df113.append(d)
    idx+=1

df1=df
df1['id']=1
distance_df113['id']=113  
distance_df113

# bringing together results from all participants
na_results=pd.concat([distance_df113,distance_df114,distance_df119,distance_df121,distance_df129,
                      distance_df130,distance_df131,distance_df134,distance_df135,distance_df137,], ignore_index=True)
na_results['r']=np.sqrt(na_results['R-Squared'])
na_results

# arrange by distance
na_dist1=na_results.loc[na_results['distance']==1].sort_values('R-Squared')

# mean and confidence interval
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

mean_confidence_interval(na_dist1['R-Squared'])

# plotting r squared results for random forest vs elastic net for each person

fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=False, sharey=True)
fig.suptitle('', fontsize = 15)

i=range(2)
j=range(5)


pid=1
ids=[113,114,119,121,129,130,131,134,135,137]

for x in i:
    for y in j:
        data=na_results.loc[na_results["id"]==ids[pid-1]][['distance','R-Squared']]
        data=data.rename(columns = {'distance':'Distance'})
        ax=sns.lineplot(ax=axes[x, y], data=data,x='Distance',y="R-Squared", label="Random forest",color='#1f77b4' )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(''.join(map(str, ['Participant ',pid])))
        pid+=1


pid=1  
for x in i:
    for y in j:
        data=enna_results.loc[enna_results["id"]==ids[pid-1]][['distance','R-Squared']]
        data=data.rename(columns = {'distance':'Distance'})
        ax=sns.lineplot(ax=axes[x, y], data=data,x='Distance',y="R-Squared", label="Elastic net" ,linestyle='--',color='#1f77b4')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pid+=1
        
        
plt.legend()

plt.savefig('enrf_na_r2.pdf')

# plotting RMSE results for negative vs positive affect for each person
fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=False, sharey=True)
fig.suptitle('', fontsize = 15)

i=range(2)
j=range(5)


pid=1
ids=[113,114,119,121,129,130,131,134,135,137]

for x in i:
    for y in j:
        data=na_results.loc[na_results["id"]==ids[pid-1]][['distance','RMSE']]
        data=data.rename(columns = {'distance':'Distance'})
        ax=sns.lineplot(ax=axes[x, y], data=data,x='Distance',y="RMSE", label="Negative affect" )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(''.join(map(str, ['Participant ',pid])))
        pid+=1
        
pid=1
for x in i:
    for y in j:
        data=pa_results.loc[pa_results["id"]==ids[pid-1]][['distance','RMSE']]
        data=data.rename(columns = {'distance':'Distance'})
        ax=sns.lineplot(ax=axes[x, y], data=data,x='Distance',y="RMSE", label="Positive affect" )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pid+=1
        
plt.savefig('napa_rmse.pdf')

# plotting actual vs predicted scores for each person
na_vals=pd.concat([df1.loc[df1['distance']==1],df2.loc[df2['distance']==1],df3.loc[df3['distance']==1],
                   df4.loc[df4['distance']==1],df5.loc[df5['distance']==1],
                   df6.loc[df6['distance']==1],df7.loc[df7['distance']==1],
                   df8.loc[df8['distance']==1],df9.loc[df9['distance']==1],df10.loc[df10['distance']==1]], ignore_index=True)

fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=False, sharey=True)
fig.suptitle('', fontsize = 15)

i=range(2)
j=range(5)


pid=1
for x in i:
    for y in j:
        data=na_vals.loc[na_vals["id"]==pid][['actual', 'predicted','Unnamed: 0']]
        data=data. rename(columns = {'Unnamed: 0':'Time point'})
        data=data.melt('Time point', var_name='', value_name='Negative affect')
        sns.lineplot(ax=axes[x, y], data=data,x='Time point',y="Negative affect", hue='' ).set_title(''.join(map(str, ['Participant ',pid])))
        pid+=1
        
plt.savefig('na_dist1.pdf')

# feature importance

# example for one person
imp=pd.read_csv('~/Downloads/p117113_na_rfimp_2505.csv')
imp=imp.drop('Unnamed: 0',1)
m=imp.groupby('feature').mean().sort_values(['importance'], ascending=False)
m=pd.DataFrame(m)
s=imp.groupby('feature').std()
s=pd.DataFrame(s)
impdf=pd.merge(m, s, on='feature', how='outer')
impdf = impdf.rename(columns={'importance_x': 'importance', 'importance_y': 'std'})
impdf['feature']=impdf.index
impdf.index = np.arange(1, len(impdf) + 1)
df1=impdf
df1['id']=1
df1['feature']=df1['feature'].replace(['com.zhiliaoapp.musically_min'], 'TikTok_min')
df1['feature']=df1['feature'].replace(['com.instagram.android_min'], 'Instagram_min')
df1['feature']=df1['feature'].replace(['com.snapchat.android_min'], 'Snapchat_min')
df1['feature']=df1['feature'].replace(['UNIQUE_MACHASHES_number'], 'Unique_WiFi_number')
df1['feature']=df1['feature'].replace(['APPS_OPENED_number'], 'Apps_Opened_number')
df1['feature']=df1['feature'].replace(['APP_USAGE_min'], 'App_Usage_min')
df1


imp_vals=pd.concat([df1[df1.index.isin(range(6))],df2[df2.index.isin(range(6))],df3[df3.index.isin(range(6))],
                   df4[df4.index.isin(range(6))],df5[df5.index.isin(range(6))],df6[df6.index.isin(range(6))],df7[df7.index.isin(range(6))],
                   df8[df8.index.isin(range(6))],df10[df10.index.isin(range(6))]])

imp_vals

rank=imp_vals[['feature','id']]
rank['rank']=rank.index
rank=rank.rename(columns={'feature': 'Feature', 'id': 'Participant ID'})
rank

# heatmap
plt.figure(figsize=(8, 7))
rankplot = rank.pivot( 'Feature', 'Participant ID', "rank")
rankplot.index = pd.CategoricalIndex(rankplot.index, categories=list((imp_vals.feature.value_counts().sort_values(ascending=False)).index))
rankplot.sort_index(level=0, inplace=True)
sns.heatmap(rankplot,cbar_kws={"ticks":range(1,6),'label': 'Rank'},annot=True,cbar=False)

plt.savefig('na_heatmap.pdf',bbox_inches = 'tight')  

