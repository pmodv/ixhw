from numpy.core.fromnumeric import sort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np


def check_df(df):

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



    colors = ["#0101DF", "#DF0101"]

    f, axes = plt.subplots(2, 2, sharex=False, sharey=False)

    # when I examine this plot, I see the potential for an ordinal encoding of 'month' helping, not hurting, RF, in splitting data.
    # NB: turns out the above line is completely wrong ;)  
    sns.countplot('month', data=df, palette=colors,hue='y',order =list(map(lambda x: str(x+1), list(range(12)))),ax = axes[0,0])
    axes[0,0].set_title('Month Distributions \n Month of Year || 1-12', fontsize=8)
    df['month'] = pd.to_numeric(df['month'])

    # we will use target encoding for 'job' field
    # will be evaluated during CV and used for each test
    # *update: v2 of class will use tgt encoding, too


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
    
    plt.show()

    # HUUUGE imbalance in target - will need sampling strategy to make RF work properly


# only code to execute for runtime proc
try:
    df = pd.read_csv('bank-full.csv',sep=';')
except IOError as e:
    print(e)

# check df and make basic diagnostic plots
# disabled unless explicit diag proc
if __name__ == '__main__':
    check_df(df)
