import datetime
import socket
import threading
import argparse
import time
import logging
import json
import pandas as pd
import numpy as np
import torch
import model
import getData
import GenRand
import communicate as comm
from sklearn.metrics import f1_score

P = 1<<128               # big prime
batch_size = 50
lr=1e-3
parties_num = 5
epochs = 4
EPOCHS = 10*epochs+1
ip_list=['127.0.0.1','127.0.0.1','127.0.0.1','127.0.0.1','127.0.0.1']              # testing
#ip_list=['192.168.80.11','192.168.80.12','192.168.80.13','192.168.80.14','192.168.80.15']
ports = [[9011,9012,9013,9014,9015],[9021,9022,9023,9024,9025],[9031,9032,9033,9034,9035],[9041,9042,9043,9044,9045],[9051,9052,9053,9054,9055]]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-id', '--id', type=int, default=1, help='client id')
args = parser.parse_args()
args = args.__dict__
id = args['id'] - 1

group_id='968'
logger_dir = "../log/Group"+group_id
logger_name = "../log/Group"+group_id+"/Group"+group_id+"Container"+str(args['id'])+".log"
open(logger_name,'w')
formatter = '%(asctime)s -- %(filename)s[line:%(lineno)d] %(levelname)s\t%(message)s'
logging.basicConfig(format=formatter, level=logging.DEBUG)
fh = logging.FileHandler(filename=logger_name,mode='a')
logger = logging.getLogger('logger')
logger.addHandler(fh)

def update_ports():
    global ports
    for i in range(parties_num):
        for j in range(parties_num):
            ports[i][j] += 50

semaphore1 = threading.BoundedSemaphore(1)
semaphore2 = threading.BoundedSemaphore(1)
pass_num = 0
LinkedNum = 0
part_sum = []
def aggregate(ip,port):
    global part_sum
    global LinkedNum
    global pass_num

    s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_addr=(ip,port)
    s_socket.bind(local_addr)
    s_socket.listen()
    s_socket, addr = s_socket.accept()
    #logger.info("connected with addr: ", addr)

    recv_data = comm.recv_message(s_socket)
    data_array = json.loads(recv_data)
    # read-write lock
    semaphore1.acquire()
    for i in range(len(part_sum)):
        part_sum[i] = (part_sum[i] + data_array[i])%P
    LinkedNum += 1
    semaphore1.release()

    while LinkedNum<(parties_num-1):
        time.sleep(3)

    result = []
    for i in range(len(part_sum)):
        result.append(part_sum[i])
    comm.send_message(s_socket,json.dumps(result))

    # lock
    semaphore2.acquire()
    pass_num += 1
    if(pass_num==(parties_num-1)):
        LinkedNum=0
        pass_num=0
    semaphore2.release()
    #logger.info('aggregate finished')
    s_socket.close()

semaphore3 = threading.BoundedSemaphore(1)
semaphore4 = threading.BoundedSemaphore(1)
pass_num_send = 0
received_part_sum_datas=[0,0,0,0,0]
received_cnt = 0
def client_send_recv(data, ip, port):
    global received_cnt
    global received_part_sum_datas
    global pass_num_send
    
    # connect
    addr = (ip,port)
    while True:
        try:
            c_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c_socket.connect(addr)
            #logger.info("connect to addr: ", addr)
            break
        except Exception as e:
            # 访问异常的错误编号和详细信息
            #logger.info(e.args)
            #logger.info("try to connect ", addr)
            time.sleep(5)

    #send message
    comm.send_message(c_socket,json.dumps(data))

    #recv message
    recv_data = comm.recv_message(c_socket)
    data_array = json.loads(recv_data)
    
    #store data
    semaphore3.acquire()
    received_part_sum_datas[received_cnt]=data_array
    received_cnt+=1
    semaphore3.release()

    while received_cnt<(parties_num-1):
        time.sleep(3)

    #clear cnt
    semaphore4.acquire()
    pass_num_send += 1
    if(pass_num_send==(parties_num-1)):
        received_cnt = 0
        pass_num_send = 0
    semaphore4.release()

    c_socket.close()

if __name__ == '__main__':
    #sg denoise + normalization
    #getData.denoise(args['id'])          # testing
    #logger.info('denoise finished')
    
    #load file list
    logger.info(f'file loading...')
    Train=getData.load_data(args['id'],istest=False)
    logger.info(f'train samples num: {len(Train)}')
    '''
    Test=getData.load_denoise_data(args['id'],istest=True)
    logger.info(f'test samples num: {len(Test)}')
    test_batch_num = int(len(Test)/batch_size)
    gaps = batch_size-(len(Test)%batch_size)
    logger.info(f'Test len, Test batch num, gaps: {len(Test)}, {test_batch_num}, {gaps}')
    gaps_test=Test[0:gaps]
    Test=Test+gaps_test'''
    
    #model set
    class_num = [0,0,0,0,0,0,0]
    for i in range(len(Train)):
        class_num[Train[i][1]] += 1
    #logger.info('samples num: ',class_num)
    class_num = np.array(class_num,dtype=np.float64)
    average = np.sum(class_num)/7.0
    for i in range(class_num.shape[0]):
        if  class_num[i]<average:
            class_num[i] = average/class_num[i]
        else:
            class_num[i] = class_num[i]/average
    #logger.info('weights: ',class_num)
    model = model.model()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_num))                            #loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)                                          #Adam optimizer

    #train
    logger.info(f'start training')
    for i_epoch in range(EPOCHS):
        #training
        runtime_loss=0
        train_gen = getData.generate_data(Train, batch_size=batch_size)
        for i in range(int(len(Train)/batch_size)):
            inputs, label = next(train_gen)
            inputs = torch.Tensor(np.array(inputs))
            inputs = inputs.unsqueeze(dim=2)
            label = torch.Tensor(np.array(label))
            y_predict = model(inputs)
            loss = criterion(y_predict, label.long())
            runtime_loss+=loss.item()
            try:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f'{e.args}')

        logger.info(f'Epoch{i_epoch+1}  loss:{round(runtime_loss,3)}  {datetime.datetime.now()}')
    
        '''
        # aggregation
        if (i_epoch) % epochs==0:
            
            logger.info(f'aggregation start: {datetime.datetime.now()}')
            #send model parameters
            send_data, len_list, big_len_list = comm.tensor2array(model)
            data_list = GenRand.rand_list(send_data,parties_num)
            #logger.info(f'random list: {len(data_list)}, {data_list[0].shape}, {data_list}')

            #communication
            client_threads_list=[]
            for i in range(parties_num):
                if i != id:
                    #logger.info(f'connect to {(ip_list[i], ports[i][id])}')
                    client_threads_list.append(threading.Thread(target=client_send_recv, args=(data_list[i], ip_list[i], ports[i][id])))
                else:
                    part_sum = data_list[i]
            for thread in client_threads_list:
                thread.start()

            server_threads_list=[]
            for i in range(parties_num):
                if i != id:
                    #logger.info(f'connect to {(ip_list[id], ports[id][i])}')
                    server_threads_list.append(threading.Thread(target=aggregate, args=(ip_list[id], ports[id][i])))
            for thread in server_threads_list:
                thread.start()
            
            for thread in server_threads_list:
                thread.join()
            for thread in client_threads_list:
                thread.join()
            
            received_datas_sum = []
            for j in range(len(part_sum)):
                tmp_data = part_sum[j]
                for i in range(parties_num-1):
                    tmp_data = (tmp_data + received_part_sum_datas[i][j])%P
                received_datas_sum.append(tmp_data)
                
            #update model parameters
            #print('print aggregated model: ',received_datas_sum)
            model = comm.array2tensor(received_datas_sum, model, parties_num, len_list, big_len_list)
            comm.tensor2array(model)
            logger.info(f'aggregation end: {datetime.datetime.now()}')
            update_ports()
            
            #save model
            torch.save(model.state_dict(),'../log/Group968/model_id'+str(args['id'])+'_latest.pth')
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
            logger.info(f'   ')
            logger.info(f'+++++++++++++++++++++++++++++++++++++++++++++++')
            logger.info(f'+   test the aggregated model on validation set')
            logger.info(f'+++++++++++++++++++++++++++++++++++++++++++++++')
            logger.info(f'macro_f1_score: {macro_f1_score}')
            logger.info(f'confusion matrix:\n {metrics}')
            logger.info(f'+++++++++++++++++++++++++++++++++++++++++++++++')
            logger.info(f'   ')
    
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
    #logger.info(Y_list)
    final = pd.DataFrame(index=name, data=[file_names,Y_list,P_list])
    final.to_csv('../log/Group968/predict'+str(args['id'])+'.csv',encoding='gbk')

    logger.info(f'FL end.......')
    '''