from numpy.lib.npyio import _savez_compressed_dispatcher
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


# reminder to use tabnet for next model with embeddings... 
from pytorch_tabnet.tab_model import TabNetClassifier


from xgboost import XGBClassifier


from ixis_diag import *

# get basic OHE done for df instance
# main difference is including month as ordinal vs tgt encoding
class my_df:

    def __init__(self, df):

        # gotta watch pointing to same objection - use copy
        self.df = pd.DataFrame(df.copy())
        
        
    def ohe(self):
        # OHE for following fields: 'marital','default', 'housing', 'loan', 'contact'
        ohe_l = ['marital','default','housing','loan','contact','poutcome']

        # Apply one-hot encoder to each column with categorical data
        self.OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.df_OH = pd.DataFrame(self.OH_encoder.fit_transform(df[ohe_l]))

        # One-hot encoding removed index; put it back
        self.df_OH.index = self.df.index

        # rename columns for new df_OH 
        self.df_OH.columns = ['married','single','divorced','def_no','def_yes','house_yes','house_no','loan_no','loan_yes',\
            'contact_unk','contact_cell','contact_tele','po_unk','po_fail','po_other','po_success']

        # Remove categorical columns from original dataframe(will replace with one-hot encoding)
        self.df_no_cat = self.df.drop(ohe_l, axis=1)

        # Add one-hot encoded columns to numerical features
        self.df_OH = pd.concat([self.df_no_cat, self.df_OH], axis=1)

        m1 = {'yes':'1','no':'0'}
        self.df_OH['y'] = pd.to_numeric(self.df_OH['y'].map(m1))


    # set attributes; don't use getters - this is python
    def set_y(self):
        self.y = self.df_OH['y']

    # set attributes; don't use getters - this is python      
    def set_X(self):
        self.X = self.df_OH.drop('y', axis = 1)

    def set_ord_code_ed(self):
        x = {'primary':'1','secondary':'2','tertiary':'3','unknown':'0'}
        self.df['education'] = pd.to_numeric(self.df['education'].map(x))
        
    def set_ord_code_month(self):
        x = {'jan':'1', 'feb':'2', 'mar':'3','apr':'4', 'may':'5', 'jun':'6','jul':'7','aug':'8','sep':'9','oct':'10','nov':'11','dec':'12'}
        self.df['month'] = pd.to_numeric(self.df['month'].map(x))

    # generic tgt encoding method
    # by-what is string arg for on which field to tgt encode
    def tgt_encode(self,by_what,train_idx):

        # this isn't haskell
        assert isinstance(by_what, str)
        # this type check was trickier than i anticipated
        assert isinstance(train_idx, (np.ndarray,np.generic))

        return self.df_OH.iloc[train_idx,:].groupby(by_what)['y'].agg('mean')
        # replace values for training data set use avg cv target encoding
        #print(means)
    
        
    
# run diagnostics from ixis_diag.py



""" model = BalancedRandomForestClassifier(n_estimators=100)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

f1 =[]
p = []
r = []
features = []

input_1 = my_df(df)

# we set month ordinals, in this current iteration (hint: it's terrible - use tgt encoding for v2)
input_1.set_ord_code_ed()
input_1.set_ord_code_month()


# build OHE df
input_1.ohe()

# ordinals

# all encoded - split into X/y

# make it so
input_1.set_X()
input_1.set_y()

X= input_1.X
y= input_1.y
 
for train_idx, test_idx in cv.split(X,y):
    X_train, X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]

    # update target encoding after shuffle/split but before fitting
    
    # replace values for training data set use avg cv target encoding
    m1 = input_1.tgt_encode('job',train_idx)
    X_train['job'] = X_train['job'].map(m1)
    X_test['job'] = X_test['job'].map(m1)
    
    
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

l_cols = X.columns
df_f_avg = pd.DataFrame(list(zip(l_cols,feature_avg)), columns=['Feature','Average_Importance'])
sorted_df = df_f_avg.sort_values(by=['Average_Importance'], ascending=False)

print(sorted_df)
sns.barplot(x='Feature', y='Average_Importance',data=sorted_df)
plt.title('RF with month using ordinal encoding')
plt.xticks(rotation = 90)
plt.show()


# version 2 - same exact setup - EXCEPT - use tgt encoding for month

f1_train = []
f1 =[]
p = []
r = []
features = []

input_2 = my_df(df)

# ordinals
# we set month ordinals, in this current iteration (hint: it's terrible - use tgt encoding for v2)
input_2.set_ord_code_ed()


# step 1 - build OHE object
input_2.ohe()

# all encoded - split into X/y

# make it so
input_2.set_X()
input_2.set_y()

# local X,y for cv split
X= input_2.X
y= input_2.y

for train_idx, test_idx in cv.split(X,y):
    X_train, X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]

    # update target encoding after shuffle/split but before fitting
    
    # replace values for training data set use avg cv target encoding
    m1 = input_2.tgt_encode('job',train_idx)
    X_train['job'] = X_train['job'].map(m1)
    X_test['job'] = X_test['job'].map(m1)

    m2 = input_2.tgt_encode('month',train_idx)
    X_train['month'] = X_train['month'].map(m2)
    X_test['month'] = X_test['month'].map(m2)
    
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



l_cols = X.columns
df_f_avg = pd.DataFrame(list(zip(l_cols,feature_avg)), columns=['Feature','Average_Importance'])
sorted_df = df_f_avg.sort_values(by=['Average_Importance'], ascending=False)

print(sorted_df)
sns.barplot(x='Feature', y='Average_Importance',data=sorted_df)
plt.title('RF with month using tgt encoding')
plt.xticks(rotation = 90)
plt.show()


 """
# NB: could use input 2 but being cautious, here
input_3 = my_df(df)

# ordinals
# we set month ordinals, in this current iteration (hint: it's terrible - use tgt encoding for v2)
input_3.set_ord_code_ed()


# step 1 - build OHE df
input_3.ohe()

# all encoded - split into X/y

# make it so
input_3.set_X()
input_3.set_y()

# local X,y for cv split
X= input_3.X
y= input_3.y


# keep it quick-ish
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)


best_f = 0
best_p = 0
best_r = 0
"""
for w in [1,10,25,50,90,100,1000]:
    model = XGBClassifier(scale_pos_weight = w)

    for train_idx, test_idx in cv.split(X,y):
        X_train, X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]

        # update target encoding after shuffle/split but before fitting
        
        # replace values for training data set use avg cv target encoding
        m1 = input_3.tgt_encode('job',train_idx)
        X_train['job'] = X_train['job'].map(m1)
        X_test['job'] = X_test['job'].map(m1)

        m2 = input_3.tgt_encode('month',train_idx)
        X_train['month'] = X_train['month'].map(m2)
        X_test['month'] = X_test['month'].map(m2)
        
        print(X_test['education'])
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

    # best f1 from W=10 (kinda makes sense)

"""


input_1 = my_df(df)

# we set month ordinals, in this current iteration (hint: it's terrible - use tgt encoding for v2)
input_1.set_ord_code_ed()
input_1.set_ord_code_month()


# build OHE df
input_1.ohe()

# ordinals

# all encoded - split into X/y

# make it so
input_1.set_X()
input_1.set_y()

X= input_1.X
y= input_1.y


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

f1 =[]
p = []
r = []
features = []

model = XGBClassifier(scale_pos_weight=10)
for train_idx, test_idx in cv.split(X,y):
    X_train, X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]

    # update target encoding after shuffle/split but before fitting
    
    # replace values for training data set use avg cv target encoding
    m1 = input_1.tgt_encode('job',train_idx)
    X_train['job'] = X_train['job'].map(m1)
    X_test['job'] = X_test['job'].map(m1)
    
    m2 = input_1.tgt_encode('month',train_idx)
    X_train['month'] = X_train['month'].map(m2)
    X_test['month'] = X_test['month'].map(m2)
    
    
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

l_cols = X.columns
df_f_avg = pd.DataFrame(list(zip(l_cols,feature_avg)), columns=['Feature','Average_Importance'])
sorted_df = df_f_avg.sort_values(by=['Average_Importance'], ascending=False)

print(sorted_df)
sns.barplot(x='Feature', y='Average_Importance',data=sorted_df)
plt.title('XGboost')
plt.xticks(rotation = 90)
plt.show()
