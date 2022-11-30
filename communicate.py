import numpy as np
import torch
import struct
import socket

bit_len = 23
sign_len = 3
value_len = 20
total_len = 26
gap = 1e-6
N = 4   # how many numbers in one big random

'''
input list: float
output list: quantify bignum
'''
def encode(array):
    trunc = (1<<(bit_len))-1
    en_array=[]
    big_num=0
    for i in range(len(array)):
        num = array[i]
        num = round(num/gap)
        max_int = (1<<value_len)-1
        if num <= -max_int:
            num = (7<<value_len)+1  #111 0000000001
        elif num >= max_int:
            num = max_int           #000 1111111111
        elif num >= 0:
            num = num               #000 1010101010
        else:
            num = num & trunc       #111 1010101010
        big_num = big_num | (num<<(total_len*(i%N)))
        if ((i+1)%N)==0:
            en_array.append(big_num)
            big_num=0
    if (len(array)%N)!=0:
        en_array.append(big_num)
    return en_array

'''
input list: quantify bignum
output list: float
'''
def decode(array):
    de_array=[]
    trunc = (1<<bit_len)-1
    for i in range(len(array)):
        num = array[i]
        for j in range(N):
            tmp = num & trunc
            if (tmp & (1<<(bit_len-1))) == 0:
                de_array.append(tmp*gap)
            else:
                tmp = tmp ^ (1<<(bit_len-1))
                tmp = tmp ^ ((1<<(bit_len-1))-1)
                tmp = tmp + 1
                de_array.append(-(tmp*gap))
            
            num = num>>total_len
            #print('num',hex(num))
    return de_array

def tensor2array(model):
    array=[]
    len_list=[]
    big_len_list=[]
    #flag=0
    for layer in model.parameters():
        layer_array = layer.detach().numpy()
        '''
        if flag==0:
            print('test--------',layer_array[0][0][0])
            flag=1
        '''
        array.append(layer_array.flatten().tolist())
    array_big=[]
    for i in range(len(array)):
        len_list.append(len(array[i]))
        en_array = encode(array[i])
        big_len_list.append(len(en_array))
        array_big.extend(en_array)
    return array_big,len_list,big_len_list

def array2tensor(parameters, model, parties_num, len_list, big_len_list):
    _cnt = 0
    _cnt_list=0
    #print(parameters.shape)
    for layer in model.parameters():
        with torch.no_grad():
            tmp_array = parameters[_cnt : _cnt+big_len_list[_cnt_list]]
            de_array = decode(tmp_array)
            de_array = np.array(de_array[:len_list[_cnt_list]])

            _cnt += big_len_list[_cnt_list]
            _cnt_list+=1
            de_array = de_array.reshape(list(layer.size()))
            de_array = de_array/parties_num
            tmp_var = torch.tensor(de_array.copy())
            layer[:] = tmp_var

    return model

def send_message(_socket, message):
    data_len = len(message)
    #print('send_len',data_len)
    struct_bytes = struct.pack('i', data_len)
    _socket.send(struct_bytes)
    _socket.send(message.encode())

def recv_message(_socket):
    #_socket.settimeout(5)
    struct_bytes = _socket.recv(4)
    data_len = struct.unpack('i', struct_bytes)[0]
    #print('recv_len',data_len)
    # 循环接收数据
    gap_abs = data_len % 1024
    count = data_len // 1024
    recv_data = b''
    
    #_socket.settimeout(5)
    for i in range(count):
        data = _socket.recv(1024, socket.MSG_WAITALL)
        #data = _socket.recv(1024)
        recv_data += data
    recv_data += _socket.recv(gap_abs, socket.MSG_WAITALL)
    #recv_data += _socket.recv(gap_abs)
    #recv = recv_data.decode()
    return recv_data.decode()
