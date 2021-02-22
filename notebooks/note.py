"""This script stores all methods used in the notebook."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import ppscore as pps
import plotly.express as px
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Experiment():
    """Base class to store the methods."""


    def __init__(self):
        """Initialize the class object."""


    def transform_datetime(df, col_name: str, new_col_name: str, how=None):
        """Method to transform datetime objects and output new columns."""
        if how == 'month':
          df[new_col_name] = pd.DatetimeIndex(df[col_name]).month
        elif how == 'year':
          df[new_col_name] = pd.DatetimeIndex(df[col_name]).year
        elif how == 'day':
          df[new_col_name] = pd.DatetimeIndex(df[col_name]).day
        elif how == 'hour':
          df[new_col_name] = pd.DatetimeIndex(df[col_name]).hour
        elif how == 'minute':
          df[new_col_name] = pd.DatetimeIndex(df[col_name]).minute
        elif how == 'second':
          df[new_col_name] = pd.DatetimeIndex(df[col_name]).second
        else:
          df[new_col_name] = pd.DatetimeIndex(df[col_name]).date
        return df.head(3)
    

    def memory_stats(df):
        """Method to compute memory stats for each feature."""
        return pd.DataFrame(
            df.memory_usage(deep=True),
            columns=['Memory']
        )


    def pps_heatmap(df):
        """
            Function for calculating the Predictive Power Score and plotting a heatmap
                Args:
                    Pandas DataFrame or Series object
                __________
                Returns:
                    figure
        """
        pps_mtrx = pps.matrix(df)
        pps_mtrx1 = pps_mtrx[['x', 'y', 'ppscore']].pivot(columns='x', index='y',
                                                  values='ppscore')
        plt.figure(figsize = (15, 8))
        ax = sb.heatmap(pps_mtrx1, vmin=0, vmax=1, cmap="afmhot_r", linewidths=0.5,
                        annot=True)
        ax.set_title("PPS matrix")
        ax.set_xlabel("feature")
        ax.set_ylabel("target")
        return ax
    

    def count_values(col_name: str, kind: str):
        """Method to compute and plot the values count."""
        vc = df[col_name].value_counts()
        plt.figure(figsize=(24, 8))
        vc.plot(kind=kind)
        plt.show()
        return vc
    

    def plot_boxplot(df):
        """Method to visualize outliers in the data."""
        plt.figure(figsize=(24, 8))
        sb.boxplot(data=df)
        plt.show()
    

    def plot_sunburst(dframe):
        """Method to plot sunburt chart"""
        colors = ['#BA4053', '#8FD5D1', '#EE6A27', '#EB4156', '#BC5545', '#B16EAF']
        return px.sunburst(
            dframe, path=['Gender', 'Married', 'Dependents', 'Education',
                        'SelfEmployed', 'CreditHistory', 'PropertyArea',
                        'LoanStatus', 'Purpose'], 
                    values='ApplicantIncome', color='LoanAmount',
                    color_continuous_scale=colors
        )


    def corr_heatmap(df, mask: bool):
        """Method to visualize correlation."""
        plt.figure(figsize=(24, 8))
        if mask == True:
            # Create mask
            mask = np.zeros_like(df.corr(), dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            # Generate Custom diverging cmap
            sb.heatmap(df.corr(), annot=True, cmap='cividis', linewidth=.5,
                       mask=mask)
        else:
            sb.heatmap(df.corr(), annot=True, cmap='cividis')
        return plt.show()

    
    def confusion_matrix(truth, predictions):
        """Method to calculate the confusion matrix of the model."""
        con = pd.DataFrame(
            confusion_matrix(truth, predictions),
            index=pd.MultiIndex.from_product(
                [['Actual'], ['Negative', 'Positive']]
            ),
            columns=pd.MultiIndex.from_product(
                [['Predicted'], ['Negative', 'Positive']]
            )
        )
        return con


    def param_optimization(estimator, params, x, y, cv: int):
        """Method to perform hyper-parameter optimization."""
        grd = GridSearchCV(
            estimator, params, scoring='accuracy', n_jobs=-1, cv=cv, verbose=1
        )
        grd.fit(x, y)
        print('Best Paramaters: ', (grd.best_params_))
        print('Best Score: ', (grd.best_score_))
        print('Best Estimator: ', (grd.best_estimator_))
        return


    def mean_accuracy_score(est, X, y, cv: int):
        """Method to calculate average accuracy of the model."""
        res = cross_val_score(est, X, y, cv=cv, n_jobs=-1, verbose=1,
                                scoring='accuracy')
        score = print('Average Accuracy:', (np.mean(res)))
        std = print('Average Standard Deviation:', (np.std(res)))
        return
    

    def plot_prc_figure(precision, recall, thresh):
        """Method for plotting Precision Recall Curve"""
        plt.figure(figsize=(24, 10))
        plt.plot(thresh, precision[:-1], 'r--', label='Precision')
        plt.plot(thresh, recall[:-1], 'g--', label='Recall')
        plt.title('Precision Recall Curve')
        plt.xlabel('Threshold')
        plt.legend(loc='best')
        plt.ylim([-0.5, 1.5])
        plt.show()
        return
    

    def roc_curve_plot(fpr, tpr, truth, pred, label=None):
        """Method to plot receiver operator characteristics curve."""
        roc = print('ROC Score:', roc_auc_score(truth, pred))
        plt.figure(figsize=(18, 10))
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        return roc

    
    def plot_roc_curve(fpr, tpr, thresholds):
        """Method to plot roc curve with rich visualization."""
        specs = pd.DataFrame({
            'FALSE POSITIVE RATE': fpr,
            'TRUE POSITIVE RATE': tpr
        }, index=thresholds)

        specs.index.name = "Thresholds"
        specs.columns.name = "Rate"

        fig = px.line(
            specs, title='TPR AND FPR AT EVERY THRESHOLD', width=480,
            height=640
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(range=[0, 1], constrain='domain')
        return fig.show()
    
    