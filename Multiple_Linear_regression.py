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
        head[i][j] = np.around(float(head[i][j]) - np.mean(col_temp)/ (max(col_temp)-min(col_temp)),2)

w = [0.001] * no_of_parameters #Assign randomly later
b = 0.9 #Assign randomly later
learning_rate = 0.0000001 #Use different rates after completed to check

w_hist = []
b_hist = []


iterations = 0
while(True):
    sq_err = 0
    diff_w = [0] * no_of_parameters
    diff_b = [0] * no_of_parameters

    #Implementing gradient descent 

    #print(w)
    for j in range(no_of_parameters):
        for i in range(no_of_data):
            diff_w[j] = diff_w[j] + ((((np.dot(w,head[i])) + b) - totchg[i])*((-1)*head[i][j]))
            diff_b[j] = diff_b[j] + ((((np.dot(w,head[i])) + b) - totchg[i]))
        diff_w[j] = np.around((diff_w[j] / no_of_data),2)
        diff_b[j] = np.around((diff_b[j] / no_of_data),2)

        w[j] = w[j] - (learning_rate * diff_w[j])
        b = b - (learning_rate * diff_b[j])
        
    w_hist.append(w[j])
    b_hist.append(b)


    print(w)
    iterations = iterations + 1
    if(iterations > 1000):
            break
    
print("FInal w :", w, "Final b",b)
