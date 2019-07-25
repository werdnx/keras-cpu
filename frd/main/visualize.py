import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("/Users/werdn/PycharmProjects/frd/resorces/fraud.csv", sep=',')

def graph1():
    print("start grap1")
    fig, ax = plt.subplots(1,2)
    ax[0].hist(train.TransactionAmt, 10, facecolor='red', alpha=0.5, label="TransactionAmt")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
    ax[1].hist(train.TransactionAmt, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="TransactionAmtv")
    #ax[0].set_ylim([0, 1000000])
    ax[0].set_xlabel("TransactionAmt")
    ax[0].set_ylabel("Frequency")
    ax[1].set_xlabel("TransactionAmt")
    ax[1].set_ylabel("Frequency")
    fig.suptitle("Distribution of TransactionAmt")
    print("before show")
    plt.show()
    print("after show")


def hist():
    print(np.histogram(train.TransactionAmt))

def cormtrx():
    corr = train.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()

def main():
    cormtrx()

if __name__ == "__main__":
    main()