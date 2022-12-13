import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
import pickle
import json

## Set up path and parameter
# file path
data_path = '/gpfs/projects/AMS598/Projects2022/project3'

# folder of current job
curr_dir = '/gpfs/home/tinghwu/project3'
# curr_dir = '/gpfs/projects/AMS598/class2022/tinghwu/project3'

tmp_dir = '/gpfs/scratch/tinghwu'


beta = 0.9

k = 10

loops = 5

def chunks(total_list, n):
    output  = [None] * n
    for idx, item in enumerate(total_list):
        if output[idx % n] is None:
            output[idx % n] = [item]
        else:
            output[idx % n].append(item)
    return output


# mapper
def mapper_init(filename, map_dic, total_url):
    f = open(filename, 'r')
    lines = f.readlines()[1:-1]
    for idx, line in enumerate(lines):
        curr, dest = [int(i) for i in line.split(',')]
        if not curr in map_dic:
            map_dic[curr] = [1, [dest]]
        else:
            map_dic[curr][1].append(dest)
        total_url.add(curr)
        total_url.add(dest)
    f.close()

    return map_dic, total_url


def reducer_init(filename, curr_list, map_dic, N):
    with open(filename, 'r') as openfile:
        curr_dic = json.load(openfile)

    for curr in curr_list:
        if str(curr) in curr_dic:
            if curr in map_dic:
                map_dic[curr][1] += curr_dic[str(curr)][1]
            else:
                map_dic[curr] = curr_dic[str(curr)]
                map_dic[curr][0] = 1/float(N)
    return map_dic


def mapper(map_dic):
    counts = {}
    point2list = {}
    for curr in map_dic:
        pr, dest = map_dic[curr]
        for url in dest:
            if not url in counts:
                counts[url] = float(pr) / float(len(dest))
            else:
                counts[url] += float(pr) / float(len(dest))
        point2list[curr] = dest
    return counts, point2list


def reducer(filenames, point2list, beta):
    map_dic = {}
    for filename in filenames:
        with open(filename, 'r') as openfile:
            counts = json.load(openfile)

        for count in counts:
            curr = int(count)
            if curr in point2list:
                if not curr in map_dic:
                    map_dic[curr] = [counts[count], point2list[curr]]
                else:
                    map_dic[curr][0] += counts[count]

    for url in map_dic:
        map_dic[url][0] *= beta
        map_dic[url][0] += (1-beta) / len(point2list)

    return map_dic


def find_top_k(map_dic, k):
    top_k_list = []
    top_k_val = []
    for item in map_dic:
        if len(top_k_list) < k:
            top_k_list.append(item)
            top_k_val.append(map_dic[item][0])
        else:
            curr_min = min(top_k_val)
            curr_val = map_dic[item][0]
            if curr_val > curr_min:
                idx = top_k_val.index(curr_min)
                top_k_val[idx] = curr_val
                top_k_list[idx] = item

    return top_k_list, top_k_val


def main():
    # time_0 = time.time()
    comm = MPI.COMM_WORLD

    # Distribute processed files
    if comm.rank == 0:
        total_list = os.listdir(data_path)
        total_list = [os.path.join(data_path, i) for i in total_list if len(i) == 3]
        data_list = chunks(total_list, comm.size)
    else:
        data_list = None

    data_list = comm.scatter(data_list, root=0)
    comm.Barrier()
    print(comm.rank, len(data_list))

    # Run the initial mapper
    map_dic = {}
    total_url = set()
    for datafile in data_list:
        map_dic, total_url = mapper_init(datafile, map_dic, total_url)

    init_files = os.path.join(tmp_dir, 'init_{}.json'.format(str(comm.rank)))
    with open(init_files, 'wb') as tmp_file:
        json.dump(map_dic, tmp_file, indent=4)
    
    comm.Barrier()
    total_url = comm.gather(total_url)
    init_files = comm.allgather(init_files)
    
    if comm.rank == 0:
        curr_total = set().union(*total_url)
        curr_list = chunks(curr_total, comm.size)
        N = len(curr_total)
    else:
        curr_list = None
        N = None

    curr_list = comm.scatter(curr_list, root=0)
    N = comm.bcast(N, root=0)
    print(comm.rank, len(curr_list))
    print(comm.rank, init_files)

    map_dic = {}
    for init_file in init_files:
        map_dic = reducer_init(init_file, curr_list, map_dic, N)

    comm.Barrier()
    for loop in range(loops):
        counts, point2list = mapper(map_dic)

        tmp_files = os.path.join(tmp_dir, 'tmp_{}_{}.json'.format(str(loop), str(comm.rank)))
        with open(tmp_files, 'wb') as tmp_file:
            json.dump(counts, tmp_file, indent=4)

        comm.Barrier()
        tmp_files = comm.allgather(tmp_files)
        print(comm.rank, tmp_files)

        map_dic = reducer(tmp_files, point2list, beta)


    top_k_list, top_k_val = find_top_k(map_dic, k)

    top_k_list = comm.gather(top_k_list, root=0)
    top_k_val = comm.gather(top_k_val, root=0)

    if comm.rank == 0:
        final_k_list = []
        final_k_val = []
        for k_list, k_val in zip(top_k_list, top_k_val):
            if len(final_k_list) < k:
                final_k_list = k_list
                final_k_val = k_val
            else:
                for num, val in zip(k_list, k_val):
                    curr_min = min(final_k_val)
                    if val > curr_min:
                        idx = final_k_val.index(curr_min)
                        final_k_list[idx] = num
                        final_k_val[idx] = val

        print(final_k_list, final_k_val)


    final_files = os.path.join(tmp_dir, 'final_{}.json'.format(str(comm.rank)))
    with open(final_files, 'wb') as tmp_file:
        json.dump(map_dic, tmp_file, indent=4)
    


if __name__ == '__main__':

    main()
