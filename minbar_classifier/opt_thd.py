import os,sys
import pdb

from scipy.optimize import minimize,Bounds
import numpy as np

MAIN_COMMAND = "python ../main.py "
WORK_FOLDER = "opt_workspace"
LOG_FILE = 'opt_thd.log'
RESULT_KEY = 'WIN RATIO:'

def create_fexyaml(fexyaml_orig,key,new_val):
    with open(fexyaml_orig,'r') as fr:
        with open('fex.yaml','w') as fw:
            for line in fr:
                # line = line.strip()
                id = line.find(key)
                if  id > 0:
                    line = line[:id+len(key)] + " " + str(new_val[0])
                fw.write(line+'\n')

def comp_winratio(val, csvfile,fexyaml_orig ):
    # args = [csvfile,fexyaml_orig]
    # csvfile = args[0]
    # fexyaml_orig = args[1]
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    create_fexyaml(fexyaml_orig,"RETURN_THRESHOLD:", val)
    cmd = MAIN_COMMAND + csvfile + " fex.yaml >& " + LOG_FILE
    os.system("bash -c '{}'".format(cmd))
    # pdb.set_trace()
    wr=[]
    with open(LOG_FILE,'r') as f:
        for line in f:
            line = line.strip()
            id = line.find(RESULT_KEY)
            if id > 0:
                res = line[id+len(RESULT_KEY):]
                res = float(res)
                wr.append(res)

    print("Current result: ",val,wr)
    return -wr[0]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python {} <csv_file> <fex.yaml> <init_val>".format(sys.argv[0]))
        sys.exit()
    if not os.path.exists(WORK_FOLDER):
        os.mkdir(WORK_FOLDER)
    os.chdir(WORK_FOLDER)
    x0 = np.array([0.003])

    if len(sys.argv)>=4:
        x0[0] = float(sys.argv[3])
        print("init val: ", x0[0])
    # res = minimize(comp_winratio, x0, args=("../" + sys.argv[1], "../" + sys.argv[2]),bounds=Bounds(0.001,0.005),
    #                method='nelder-mead', options={'xatol': 1e-4, 'disp': True})
    res = minimize(comp_winratio, x0, args=("../" + sys.argv[1], "../" + sys.argv[2]), bounds=[(0.001,0.008)],
                   method='Powell', options={'xtol': 1e-3, 'disp': True})
    print("Best x: ",res.x)
