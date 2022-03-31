## for data
import time
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from lime import lime_tabular

#a method for classification method
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"

def classification():
    print("Please Note: this option, you not need give any input, it's just for learn how it works.")

    dtf = pd.read_csv('train.csv') #got in Kaggle
    dtf.head()

    '''
    Recognize whether a column is numerical or categorical.
    :parameter
    :param dtf: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
    :return
      "cat" if the column is categorical or "num" otherwise
    '''

    dic_cols = {col:utils_recognize_type(dtf, col, max_cat=20) for col in dtf.columns}
    
    heatmap = dtf.isnull()
    for k,v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    
    sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
    plt.savefig("plt line 49.png")
    #plt.show()
    
    #print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")
    print("[1;37;40m Categerocial ", "[1;30;41m Numeric ", "[1;30;47m NaN ")

    dtf = dtf.set_index("PassengerId")
    dtf = dtf.rename(columns={"Survived":"Y"})

    y = "Y"
    
    ax = dtf[y].value_counts().sort_values().plot(kind="barh")
    totals= []
    
    for i in ax.patches:
        totals.append(i.get_width())
    total = sum(totals)
    
    for i in ax.patches:
        ax.text(i.get_width()+.3, i.get_y()+.20, 
        str(round((i.get_width()/total)*100, 2))+'%', 
        fontsize=10, color='black')
    
    ax.grid(axis="x")
    
    plt.suptitle(y, fontsize=20)
    plt.savefig('plt 2 line 74.png')
    #plt.show()

    x = "Age"
    
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    
    fig.suptitle(x, fontsize=20)
    
    ### distribution
    
    ax[0].title.set_text('distribution')
    variable = dtf[x].fillna(dtf[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[ (variable > breaks[0]) & (variable < 
                    breaks[10]) ]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
    des = dtf[x].describe()
    ax[0].axvline(des["25%"], ls='--')
    ax[0].axvline(des["mean"], ls='--')
    ax[0].axvline(des["75%"], ls='--')
    ax[0].grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
    ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    
    ### boxplot 
    
    ax[1].title.set_text('outliers (log scale)')
    tmp_dtf = pd.DataFrame(dtf[x])
    tmp_dtf[x] = np.log(tmp_dtf[x])
    tmp_dtf.boxplot(column=x, ax=ax[1])
    #plt.show()
    plt.savefig("plt3 line 107.png")


    # cat, num = "Y", "Age"
    
    # fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False)
    # fig.suptitle(x+"   vs   "+y, fontsize=20)
    
    # ### distribution
    # ax[0].title.set_text('density')
    # for i in dtf[cat].unique():
    #     sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
    # ax[0].grid(True)

    # ### stacked    
    # ax[1].title.set_text('bins')
    # breaks = np.quantile(dtf[num], q=np.linspace(0,1,11))
    # tmp = dtf.groupby([cat, pd.cut(dtf[num], breaks, duplicates='drop')]).size().unstack().T
    # tmp = tmp[dtf[cat].unique()]
    # tmp["tot"] = tmp.sum(axis=1)
    
    # for col in tmp.drop("tot", axis=1).columns:
    #  tmp[col] = tmp[col] / tmp["tot"]
    # tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
    # ### boxplot   
    # ax[2].title.set_text('outliers')
    # sns.catplot(x=cat, y=num, data=dtf, kind="box", ax=ax[2])
    # ax[2].grid(True)

    # #plt.show()
    # plt.savefig("plt4 line 136.png")

    cat, num = "Y", "Age"
    
    model = smf.ols(num+' ~ '+cat, data=dtf).fit()
    table = sm.stats.anova_lm(model)
    p = table["PR(>F)"][0]
    coeff, p = None, round(p, 3)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")

    x, y = "Sex", "Y"
    
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    fig.suptitle(x+"   vs   "+y, fontsize=20)
    
    ### count
    ax[0].title.set_text('count')
    order = dtf.groupby(x)[y].count().index.tolist()
    sns.catplot(x=x, hue=y, data=dtf, kind='count', order=order, ax=ax[0])
    ax[0].grid(True)
    
    ### percentage
    ax[1].title.set_text('percentage')
    a = dtf.groupby(x)[y].count().reset_index()
    a = a.rename(columns={y:"tot"})
    b = dtf.groupby([x,y])[y].count()
    
    b = b.rename(columns={y:0}).reset_index()
    b = b.merge(a, how="left")
    b["%"] = b[0] / b["tot"] *100
    sns.barplot(x=x, y="%", hue=y, data=b,
            ax=ax[1]).get_legend().remove()
    ax[1].grid(True)
    plt.show()
    plt.savefig("plt5 line 173.png")

def main():
    classification()


if __name__ == '__main__':
    main()