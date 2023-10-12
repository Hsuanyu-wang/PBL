import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from os import walk
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
 
train_input=[]
train_input_std=[]
train_output=[]
folder_name=['stable','unstable']
i = 0
for folder in folder_name:
    path = 'Data/'+str(folder)+'/'
    for root,dirs,files in walk(path):
        for f in files:
            filename = path + f
            print(filename)
            acc = genfromtxt(filename, delimiter=',')
            acc = acc[:,1].tolist()
            train_input.append(acc[60000:80000])
            train_input_std.append(np.std(acc))
            
            if folder == 'unstable':
                train_output.append(1)
                title = 'Original Signal With Chatter #'
                saved_file_name = 'Fig/Original/unstable_'
            if folder == 'stable':
                train_output.append(0)
                title = 'Original Signal Without Chatter #'
                saved_file_name = 'Fig/Original/stable_'
                
            plt.figure(figsize=(7,4))
            plt.plot(acc,'b-',lw=1)
            plt.title(title + str(i+1))
            plt.xlabel('Samples')
            plt.ylabel('Acceleration')
            plt.savefig(saved_file_name + str(i+1) + '.png')
            plt.show()
            i = i + 1
                
train_input = np.array(train_input_std)
train_output = np.array(train_output)
 
scaler = MinMaxScaler(feature_range=(0,1))
train_input=scaler.fit_transform(train_input.reshape(-1,1))
 
loo = LeaveOneOut()
model = MLPClassifier(max_iter=500, batch_size=1, solver='adam')
 
y_pred = cross_val_predict(model, train_input, train_output, cv=loo)
y_true = train_output
 
print('Prediction: \t', y_pred)
print('Ground Truth: \t',y_true)
 
 
cf_m = confusion_matrix(y_true, y_pred)
print('Confusion Matrix: \n', cf_m)
 
tn, fp, fn, tp = cf_m.ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
print('Accuracy: ', accuracy)
