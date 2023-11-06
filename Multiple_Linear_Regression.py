import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

# np.set_printoptions(suppress=True,
#    formatter={'float_kind':'{:16.3f}'.format}, linewidth=130)

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
       



#Univariate Cost function(J) = wx + b
#Multivariate Cost function(J) = w1x1 + w2x2 + ... + wnxn + b
#print(head)

w = [0.9] * no_of_parameters #Assign randomly later
b = 0.1 #Assign randomly later
learning_rate = 0.1 #Use different rates after completed to check

w_hist = []
b_hist = []


iterations = 0
while(iterations < 100):
    sq_err = []
    diff_w = [0] * no_of_parameters
    diff_b = [0] * no_of_parameters

    #Implementing gradient descent 
    y_pre = [0] * no_of_data
    #print(w)
    i = 0
    j = 0
    for j in range(no_of_parameters):
        for i in range(no_of_data):
            y_pre[j] = np.dot(w,head[i]) + b - totchg[i]
            diff_w[j] = diff_w[j] + (y_pre[j])*((-1)*head[i][j])
            diff_b[j] = diff_b[j] + y_pre[j]
        diff_w[j] = (diff_w[j] / no_of_data)
        diff_b[j] = (diff_b[j] / no_of_data)
        w[j] = w[j] - (learning_rate * diff_w[j])
        b = b - (learning_rate * diff_b[j])
        
        w_hist.append(w[j])
        b_hist.append(b)

        
        iterations = iterations + 1



        

print("FInal w :", w, "Final b",b)
print(head[1])
print("2nd set" , np.dot(w,head[1]) + b)


