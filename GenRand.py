import random
P = 1<<128

def rand_list(data,parties_num):
    # generate shares
    data_size = len(data)
    data_list=[]
    #random_local
    for i in range(parties_num-1):
        r_list = []
        for j in range(data_size):
            r = random.getrandbits(128)
            r_list.append(r)
            data[j] = (data[j]-r)%P
            if data[j]<0:
                data[j]+=P
        data_list.append(r_list)
    data_list.append(data)

    return data_list

def merge_rand_list(data):
    data_list=[]
    for i in range(len(data[0])):
        tmp_data=0
        for j in range(len(data)):
            tmp_data = (tmp_data + data[j][i])%P
        data_list.append(tmp_data)
    return data_list
'''
a = [1234,753753,3573537,35725,5678,86373]
print(a)
a_list=rand_list(a,2)
print(a_list)
b=merge_rand_list(a_list)
print(b)
'''
