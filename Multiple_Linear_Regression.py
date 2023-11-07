import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd



df = pd.read_csv("C:\\Users\\risha\\Downloads\\linear_regression_dataset.csv")
data = df.fillna(0)
#pd.data.replace([np.nan],0)
to_find = 'TOTCHG'
no_of_data = len(data.axes[0]) 
no_of_parameters = len(data.axes[1]) - 1
labels = data.axes[1].tolist()
labels.remove(to_find)


#Defining x values
head = []

for i in range(1,no_of_data+1):
    head.append(data.iloc[i-1].tolist())
    del head[i-1][data.columns.get_loc(to_find)]  #data.columns.get_loc(to_find)



#List of y actual
totchg = data[to_find].tolist()




j = 0
#Normalixe data
for i in range(len(head)):
    for j in range(len(labels)):
        col_temp = data[labels[j]].tolist()
        head[i][j] = (head[i][j] - np.mean(col_temp))/ (max(col_temp)-min(col_temp))
       

tot_mean = np.mean(totchg)
for i in range(len(totchg)):

    totchg[i] = (totchg[i] - tot_mean) / (max(totchg) - (min(totchg)))


#Univariate Cost function(J) = wx + b
#Multivariate Cost function(J) = w1x1 + w2x2 + ... + wnxn + b
#print(head[0])

w = [0.05] * no_of_parameters #Assign randomly later
b = 0.0003 #Assign randomly later
learning_rate = 0.099 #Use different rates after completed to check

w_hist = []
b_hist = []


iterations = 0
while(iterations < 100):
    sq_err = []
    diff_w = [0] * no_of_parameters
    diff_b = [0] * no_of_parameters

    #Implementing gradient descent 
    y_pre = [0]*no_of_parameters
    #print(w)
    for j in range(no_of_parameters):
        for i in range(no_of_data):
            y_pre[j] = np.dot(w,head[i]) + b - totchg[i]
            diff_w[j] = diff_w[j] + (y_pre[j])*((-1)*head[i][j])
            diff_b[j] = diff_b[j] + y_pre[j]
        diff_w[j] = (diff_w[j] / no_of_data)
        diff_b[j] = (diff_b[j] / no_of_data)
        
        w_hist.append(w[j])
        b_hist.append(b)

        

    for j in range(no_of_parameters):
        w[j] = w[j] - (learning_rate * diff_w[j])
        b = b - (learning_rate * diff_b[j])

    
    print(w)
    iterations = iterations + 1



        
c = 0
print("FInal w :", w, "Final b",b)
for i in range(no_of_data):
    if((np.dot(w,head[i]) + b) - totchg[i] < 0.01):
        c=c+1
print("Accuracy=" , 100*(c/no_of_data))



