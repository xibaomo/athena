from enum import IntEnum
OPEN_KEY = "<OPEN>"
HIGH_KEY = "<HIGH>"
LOW_KEY  = "<LOW>"

class State(IntEnum):
    LIVE = 0,
    TP = 1,
    SL = 2,
    NONE=3

def count_subarr(arr,sub_arr):
    occ = 0
    sub_len = len(sub_arr)
    for i in range(len(arr)-len(sub_arr)):
        k=0
        for j in range(len(sub_arr)):
            if arr[i+j] == sub_arr[j]:
                k+=1
        if k==sub_len:
            occ+=1
    return occ