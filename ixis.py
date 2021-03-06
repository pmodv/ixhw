from numpy.core.fromnumeric import sort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from imblearn.ensemble import BalancedRandomForestClassifier


# ran out of time for tabnet...  bummer
from pytorch_tabnet.tab_model import TabNetClassifier


from xgboost import XGBClassifier

try:
    df = pd.read_csv('bank-full.csv',sep=';')
except IOError as e:
    print(e)

# let's hunt for missing data

print('is there a NaN in the table?',df.isnull().values.any())

# check again
nan_rows = df[df.isnull().any(1)]
print('nan rows',nan_rows)
# solid!

# start with categorical data
# count unique values in each category to get sense of cardinality


# check types and examine any non-uniform types (object type, specifically)
print(df.dtypes)


# make list of all object columns for deeper inspection
list_cols = list(df.select_dtypes(['object']))

# nothing weird, here
[ print(c,l) for c,l in zip(list_cols,list(map(lambda x: df[x].unique(), list_cols)))]


# cardinality
[ print(c,l) for c,l in zip(list_cols,list(map(lambda x: len(df[x].unique()), list_cols)))]

# we can use OHE for cat vars, but let's look at data distributions to anticipate its effect on RF

# look at rel freq for categoricals
[ print(c,l) for c,l in zip(list_cols,list(map(lambda x: df.value_counts([x],normalize=True),list_cols)))]


#  2 corner cases:  months:  december! << 1%  || default' field  ~1% have defaulted!
#  the seasonal aspect of the month data makes OHE unaccpetable and gives ordinal representation a shot for RF

x = {'jan':'1', 'feb':'2', 'mar':'3','apr':'4', 'may':'5', 'jun':'6','jul':'7','aug':'8','sep':'9','oct':'10','nov':'11','dec':'12'}

df['month'] = df['month'].map(x)


colors = ["#0101DF", "#DF0101"]

f, axes = plt.subplots(2, 2, sharex=False, sharey=False)

# when I examine this plot, I see the potential for an ordinal encoding of 'month' helping, not hurting, RF, in splitting data.
sns.countplot('month', data=df, palette=colors,hue='y',order =list(map(lambda x: str(x+1), list(range(12)))),ax = axes[0,0])
axes[0,0].set_title('Month Distributions \n Month of Year || 1-12', fontsize=8)
df['month'] = pd.to_numeric(df['month'])

# we will use target encoding for 'job' field
# will be evaluated during CV and used for each test



# our other ordinal candidate:  education
x = {'primary':'1','secondary':'2','tertiary':'3','unknown':'0'}
df['education'] = df['education'].map(x)

# when I examine this plot, I see the potential for an ordinal encoding of 'education' helping, not hurting, RF, in splitting data.
sns.countplot('education', data=df, palette=colors,hue='y',order =list(map(lambda x: str(x), list(range(4)))),ax = axes[0,1])
axes[0,1].set_title('Education Distributions  \n Education Level || 0-3', fontsize=8)
df['education'] = pd.to_numeric(df['education'])

# interesting split potential for poutcome... will leave for next round
sns.countplot('poutcome',data=df, palette=colors,hue='y',ax = axes[1,0])
axes[1,0].set_title('pout ', fontsize=8)


# examine distribution of target variable for imbalance

colors = ["#0101DF", "#DF0101"]

sns.countplot('y', data=df, palette=colors,ax = axes[1,1])
axes[1,1].set_title('Target Distributions \n (No: Declined || Yes: Approved)', fontsize=8)


#plt.show(block=True)

# job
sns.countplot('job', data=df, palette=colors,hue='y')
plt.title('Job Distributions \n Job', fontsize=8)
plt.xticks(rotation=45)
#plt.show()

# HUUUGE imbalance in target - will need sampling strategy to make RF work properly


# OHE for following fields: 'marital','default', 'housing', 'loan', 'contact'
ohe_l = ['marital','default','housing','loan','contact','poutcome']

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
df_OH = pd.DataFrame(OH_encoder.fit_transform(df[ohe_l]))

# One-hot encoding removed index; put it back
df_OH.index = df.index

# rename columns for new df_OH 
df_OH.columns = ['married','single','divorced','def_no','def_yes','house_yes','house_no','loan_no','loan_yes',\
    'contact_unk','contact_cell','contact_tele','po_unk','po_fail','po_other','po_success']

# Remove categorical columns from original dataframe(will replace with one-hot encoding)
df_no_cat = df.drop(ohe_l, axis=1)

# Add one-hot encoded columns to numerical features
df_OH = pd.concat([df_no_cat, df_OH], axis=1)



m1 = {'yes':'1','no':'0'}
df_OH['y'] = pd.to_numeric(df_OH['y'].map(m1))

y = df_OH['y']
X = df_OH.drop('y', axis = 1)

# we're FINALLY ready to do RF - first, we must address imbalanced data
# us imblearn and repeated stratified cv

model = BalancedRandomForestClassifier(n_estimators=100)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

f1_train = []
f1 =[]
p = []
r = []
features = []
for train_idx, test_idx in cv.split(X,y):
    X_train, X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]

    # update target encoding after shuffle/split but before fitting
    # use df_OH for this :)
    means = df_OH.iloc[train_idx,:].groupby('job')['y'].agg('mean')
    # replace values for training data set use avg cv target encoding
    #print(means)
    
    X_train['job'] = X_train['job'].map(means)
    X_test['job'] = X_test['job'].map(means)
    
    model.fit(X_train,y_train)
    
    y_pred1 = model.predict(X_test)
    f1.append((f1_score(y_test, y_pred1)))
    p.append((precision_score(y_test, y_pred1)))
    r.append((recall_score(y_test, y_pred1)))


    importance = model.feature_importances_
    
    # capture feature importance
    features.append(importance)
    
# imbalanced data tricks not really working...
print(mean(f1),mean(p),mean(r))

# get average feature importance...

feature_mat = np.array(features)
feature_avg = np.mean(feature_mat, axis = 0)

# xgbc

l_cols = X.columns
df_f_avg = pd.DataFrame(list(zip(l_cols,feature_avg)), columns=['Feature','Average_Importance'])
sorted_df = df_f_avg.sort_values(by=['Average_Importance'], ascending=False)

print(sorted_df)
sns.barplot(x='Feature', y='Average_Importance',data=sorted_df)
plt.xticks(rotation = 90)
#plt.show()

y = df_OH['y']
X = df_OH.drop('y', axis = 1)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)


best_f = 0
best_p = 0
best_r = 0
features = []
for w in [1,10,25,50,90,100,1000]:
    model = XGBClassifier(scale_pos_weight = w)

    for train_idx, test_idx in cv.split(X,y):
        X_train, X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        # update target encoding after shuffle/split but before fitting
        # use df_OH for this :)
        means = df_OH.iloc[train_idx,:].groupby('job')['y'].agg('mean')
        # replace values for training data set use avg cv target encoding
        #print(means)
        
        X_train['job'] = X_train['job'].map(means)
        X_test['job'] = X_test['job'].map(means)
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        if ((f1_score(y_test, y_pred)) > best_f):
            best_f = f1_score(y_test, y_pred)
            best_f_para = {'W':w}
        if (precision_score(y_test, y_pred)) > best_p:
            best_p = precision_score(y_test, y_pred)
            best_p_para = {'W':w}
        if (recall_score(y_test, y_pred)) > best_r:
            best_r = recall_score(y_test, y_pred)
            best_r_para = {'W':w}
        
    # imbalanced data tricks not really working...
    print(best_f,best_p,best_r)
    print(best_f_para,best_p_para,best_r_para)

    # better precision but recall dropped
    # let's encode months using OHE


