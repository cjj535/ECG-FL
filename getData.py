import os
import numpy as np
import scipy.io as scio
import scipy
import random
from sklearn import preprocessing

# 加载数据集并进行预处理
DX_CODE=['164889003','164890007','713422000','426177001','426783006','427084000','426761007']
def read_hea(file):     #打开hea目标文件
    file_contents=file.readlines()                    #按行读取全部内容
    for content in file_contents:     #逐行读取
        if  'Dx' in content:          #检查包含pH的那行数据
            content=content.replace('#Dx:','')
            content=content.strip()
            dxs=content.split(',')
            res =set()
            flag=0
            for dx in dxs:
                if dx in DX_CODE:
                    res.add(DX_CODE.index(dx))
                    flag=1
            if flag == 0:
                print("error")
            return res,len(res)

def load_denoise_data(id,istest=False):
    data_list=[]
    if istest == True:
        work_dir='../data/Group968/Container'+str(id)+'/test'
    else:
        work_dir='../data/Group968/Container'+str(id)+'/train'
    for file in os.listdir(work_dir):
        if file.endswith('_d.mat'):
            file_info=[]
            filename = file[:-6]
            mat_path = os.path.join(work_dir,filename+"_d.mat")
            file_info.append(mat_path)
            
            hea_path = os.path.join(work_dir,filename+".hea")
            dx,lendx = read_hea(open(hea_path,'r'))
            if lendx==1:
                file_info.append(dx.pop())
            else:
                print("error in "+hea_path)
            data_list.append(file_info)
    
    random.shuffle(data_list)
    return data_list

def load_data(id,istest=False):
    data_list=[]
    if istest == True:
        work_dir='../data/Group968/Container'+str(id)+'/test'
    else:
        work_dir='../data/Group968/Container'+str(id)+'/train'
    for file in os.listdir(work_dir):
        if file.endswith('_d.mat'):
            continue
        if file.endswith('.mat'):
            file_info=[]
            filename = file[:-4]
            mat_path = os.path.join(work_dir,filename+".mat")
            file_info.append(mat_path)
            
            hea_path = os.path.join(work_dir,filename+".hea")
            dx,lendx = read_hea(open(hea_path,'r'))
            if lendx==1:
                file_info.append(dx.pop())
            else:
                print("error in "+hea_path)
            data_list.append(file_info)
    
    random.shuffle(data_list)
    return data_list

def generate_data(data_list,batch_size=50,is_shuffle=True):
    # shuffle
    if is_shuffle==True:
        random.shuffle(data_list)
    # load data
    cnt = 0
    while (cnt+batch_size) <= len(data_list):
        X=[]
        Y=[]
        for i in range(batch_size):
            data = scio.loadmat(data_list[cnt+i][0])
            data = data['val']
            data = np.array(data)
            X.append(data)
            Y.append(data_list[cnt+i][1])
        # yield
        yield X,Y
        cnt += batch_size

def denoise(id):
    dir='/data/Group968/Container'+str(id)+'/train'
    for file in os.listdir(dir):
        if file.endswith('_d.mat'):
            continue
        if file.endswith('.mat'):
            filename = file[:-4]
            src_path = os.path.join(dir,filename+".mat")
            try:
                data = scio.loadmat(src_path)
                data = data['val']
                data = np.array([scipy.signal.savgol_filter(preprocessing.scale(sig),31,8) for sig in data])
                dst_path = os.path.join(dir,filename+"_d.mat")
                scio.savemat(dst_path, {'val':data})
            except Exception as e:
                print(f'cannot load {src_path}')
                print(f'{e.args}')

    dir='/data/Group968/Container'+str(id)+'/test'
    for file in os.listdir(dir):
        if file.endswith('_d.mat'):
            continue
        if file.endswith('.mat'):
            filename = file[:-4]
            src_path = os.path.join(dir,filename+".mat")
            try:
                data = scio.loadmat(src_path)
                data = data['val']
                data = np.array([scipy.signal.savgol_filter(preprocessing.scale(sig),31,8) for sig in data])
                dst_path = os.path.join(dir,filename+"_d.mat")
                scio.savemat(dst_path, {'val':data})
            except Exception as e:
                print(f'cannot load {src_path}')
                print(f'{e.args}')
'''
def load_test(data_list):
    # load data
    for sample in data_list:
        data = scio.loadmat(sample[0])
        data = data['val']
        data = np.array(data)
        # yield
        yield data,sample[1]
'''