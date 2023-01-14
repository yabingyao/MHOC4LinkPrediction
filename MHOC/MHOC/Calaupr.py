#!/usr/bin/env python
# coding: utf-8

# In[50]:


from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def points(y_test, y_proba, space = 'PR'):
    
    '''
    calculates the points in ROC and PR space
    
    input:
    y_test  (array)  = class/label of the test set 
    y_proba (array)  = probability of the predicted class/label
    
    space options:
    'ROC' for Reciever Operator Charactersics
    'PR'  for Precision-Recall (default)
    
    '''
    
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve
    
    # points in ROC space
    if space == 'ROC':

        fpr, tpr, thresholds = roc_curve(np.array(y_test), np.array(y_proba))
        Points = np.column_stack((fpr, tpr))
    
    # points in PR space+
    elif space == 'PR':
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        Points = np.column_stack((recall, precision))
    
    return Points

def auc_pr_score(y, Points_ROC, Points_PR):
    
    '''
    calculates Area Under the Curve (AUC) for Precison-Recall Curves according to Davis et al. 2006
    
    input:
    
    y           (array)      = true classes/labels
    
    Points_ROC  (*,2-array)  = Results in ROC space 
    Points_PR   (*,2-array)  = Results in PR space
    
    '''

    #1. calculating Convex Hull in ROC
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(Points_ROC)
    point_set=[]

    for simplex in hull.simplices:
        if simplex[0] in point_set:
            pass
        else:
            point_set.append(simplex[0])
        if simplex[1] in point_set:
            pass
        else:
            point_set.append(simplex[1])
    
    #2. Converting corresponding true positive rate (tpr)/false positve rate (fpr) to the confusion matrix
    Cp = sum(y)           # number of samples where true class = 1
    Cn = len(y)-Cp        # number of samples where true class = 0
    
    #creating a pd.DataFrame for the convex hull
    CONVEX_HULL = pd.DataFrame(columns=['IDX','fpr','tpr','precision','tp','tn','fp','fn'])
    
    # calculating confusion matrix for each point in the convex hull
    point_set.sort()
    
    for point in point_set:
        fpr = Points_ROC[point,0]
        tpr = Points_ROC[point,1]

        tp = tpr*Cp
        fp = fpr*Cn
        fn = Cp-tp
        tn = Cn-fp

        #3. Converting the confusion matrix to points in PR space
        recall    = tpr
        precision = tp/(fp+tp)
       
        import math
        if math.isnan(precision):
            precision = 1
        else:
            pass

        CONVEX_HULL = CONVEX_HULL.append(pd.DataFrame([[point, fpr,tpr,precision,tp,tn,fp,fn]],
                                                      columns=['IDX','fpr','tpr','precision','tp','tn','fp','fn']))
    #4. Non-linear interpolation
    
    # create pd.DataFrame for the coordinates of the non-linear interpolation in PR space
    INTERPOLATION = pd.DataFrame(columns=['x','y'])
    
    # calculate the space between convex hull according to non-linear interpolation
    for count in range(len(CONVEX_HULL)-1):
        INTERPOLATION = INTERPOLATION.append(pd.DataFrame([[CONVEX_HULL.iloc[count]['tpr'],CONVEX_HULL.iloc[count]['precision']]],columns=['x','y']))
        try:
            TP_a = int(CONVEX_HULL.iloc[count]['tp'])
            TP_b = int(CONVEX_HULL.iloc[count+1]['tp'])
            FP_a = int(CONVEX_HULL.iloc[count]['fp'])
            FP_b = int(CONVEX_HULL.iloc[count+1]['fp'])
            if TP_a==TP_b:
                term_y=0
            else:
                term_y = (FP_b-FP_a)/(TP_b-TP_a)


            x_arr = range(TP_b-TP_a)
            X=[]
            Y=[]
            for i in x_arr:
                x_coor=(TP_a+i)/Cp
                y_coor= (TP_a+i)/(TP_a+i+FP_a+(term_y*i))
                INTERPOLATION = INTERPOLATION.append(pd.DataFrame([[x_coor,y_coor]],columns=['x','y']))
        except:
            print('')
            
    #4. AUC trapezoidal integration
    from sklearn import metrics
    
    #print(INTERPOLATION.x.values)
    auc_pr = metrics.auc(INTERPOLATION['x'],INTERPOLATION['y'])

   
    
    return auc_pr, INTERPOLATION, CONVEX_HULL





def plot_auc_pr(auc_pr, Points_PR):
    
    '''
    plot interpolated curve PR_curve
    '''

    plt.figure(figsize=(10,8))
    plt.plot(Points_PR[:,0],Points_PR[:,1],color='cornflowerblue',label='Precision-Recall (AUC = %0.2f)'%auc_pr)
    plt.scatter(CONVEX_HULL['tpr'],CONVEX_HULL['precision'],color='cornflowerblue', label='Convex Hull')
    plt.plot(INTERPOLATION['x'],INTERPOLATION['y'],'k--', label='Interpolation (non-linear)')
    plt.xlim([0,1])
    plt.ylim([0,1.2])
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('Interpolated Precison-Recall Curve', fontsize=20)
    plt.legend(fontsize=15)
    
    plt.show()

    
if __name__ == '__main__':
    pass

