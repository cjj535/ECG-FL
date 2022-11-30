import numpy as np
import torch
import scipy.io as scio
import model
import getData
from sklearn.metrics import f1_score
import getData
import pandas as pd

batch_size=50

model = model.model()
model.load_state_dict(torch.load('/log/Group968/model_id1_latest.pth'))

Test=getData.load_data(1,istest=True)
print(f'test samples num: {len(Test)}')
test_batch_num = int(len(Test)/batch_size)
gaps = batch_size-(len(Test)%batch_size)
print(f'Test len, Test batch num, gaps: {len(Test)}, {test_batch_num}, {gaps}')
gaps_test=Test[0:gaps]
Test=Test+gaps_test
#f1 score
test_gen = getData.generate_data(Test, batch_size=batch_size)
metrics = np.zeros((7,7),dtype=int)
Y_list=[]
P_list=[]
with torch.no_grad():
    for i in range(int(len(Test)/batch_size)):
        inputs, label = next(test_gen)
        inputs = torch.Tensor(np.array(inputs))
        inputs = inputs.unsqueeze(dim=2)
        label = torch.Tensor(np.array(label))
        y_pred = model(inputs)
        _, predicted = torch.max(y_pred.data, dim=1)
        pred_Y = predicted.detach().numpy()
        label = label.detach().numpy()
        if i < int(len(Test)/batch_size)-1:
            for j in range(batch_size):
                Y_list.append(int(label[j]))
                P_list.append(pred_Y[j])
                metrics[int(label[j])][pred_Y[j]]+=1
        else:
            for j in range(batch_size-gaps):
                Y_list.append(int(label[j]))
                P_list.append(pred_Y[j])
                metrics[int(label[j])][pred_Y[j]]+=1

#eval.macro_f1_score(metrics=metrics)
macro_f1_score=f1_score(Y_list,P_list,average='macro')
print(f'   ')
print(f'+++++++++++++++++++++++++++++++++++++++++++++++')
print(f'+   test the aggregated model on validation set')
print(f'+++++++++++++++++++++++++++++++++++++++++++++++')
print(f'macro_f1_score: {macro_f1_score}')
print(f'confusion matrix:\n {metrics}')
print(f'+++++++++++++++++++++++++++++++++++++++++++++++')
print(f'   ')

#validation
DX_CODE=['164889003','164890007','713422000','426177001','426783006','427084000','426761007']
Y_list=[]
P_list=[]
test_gen = getData.generate_data(Test, batch_size=batch_size,is_shuffle=False)
with torch.no_grad():
    for i in range(int(len(Test)/batch_size)):
        X, Y = next(test_gen)
        X = torch.Tensor(np.array(X))
        X = X.unsqueeze(dim=2)
        #label = torch.Tensor(np.array(Y))
        y_pred = model(X)
        _, predicted = torch.max(y_pred.data, dim=1)
        pred_Y = predicted.detach().numpy()
        if i < int(len(Test)/batch_size)-1:
            for j in range(batch_size):
                Y_list.append(DX_CODE[Y[j]])
                P_list.append(DX_CODE[pred_Y[j]])
        else:
            for j in range(batch_size-gaps):
                Y_list.append(DX_CODE[Y[j]])
                P_list.append(DX_CODE[pred_Y[j]])
name=['file_name','true_label','pred_label']
file_names=[]
file_labels=[]
for i in range(batch_size*(test_batch_num+1)-gaps):
    file_names.append(Test[i][0][-13:-6])
#print(Y_list)
final = pd.DataFrame(index=name, data=[file_names,Y_list,P_list])
final.to_csv('../log/Group968/predict.csv',encoding='gbk')

print(f'FL end.......')