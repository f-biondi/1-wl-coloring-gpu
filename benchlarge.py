import json
import subprocess
import sqlite3
import os
from sys import argv

CUDA_COMMAND = "{inp} | timeout 7200 ./batched_be_aggressive {edges} {gpu}" 
CPU_COMMAND = "{inp} | timeout 7200 ./random_be" 
VALMARI_COMMAND = "{inp} | timeout 7200 ./part_ref" 
INPUT_LAW = "./webgraph to {fmt} {name}/{name}.graph"
INPUT_FILE = "{cmd} ./datasets/{name}.txt"

class Result():
    def __init__(self, done=0, nodes=0, edges=0, time=0):
        self.done = done
        self.nodes = nodes
        self.edges = edges
        self.time = time

class Gdata():
    def __init__(self, nodes=0, edges=0):
        self.nodes = nodes
        self.edges = edges

def gdata_file(name):
    res = Gdata()
    with open(f"./datasets/{name}.txt","r") as f:
        res.nodes = int(f.readline()[:-1])
        res.edges = int(f.readline()[:-1])
    return res

def gdata_law(name):
    res = Gdata()
    os.system(f"./download.sh {name}")
    gdata_txt = subprocess.run([INPUT_LAW.format(fmt="gdata", name=name)], stdout=subprocess.PIPE, shell=True).stdout.decode()
    gdata_txt = gdata_txt.split("\n")[:-1]
    res.nodes = int(gdata_txt[0])
    res.edges = int(gdata_txt[1])
    return res

def experiment(command, runs=1):
    res = Result()
    print(command)
    for _ in range(runs):
        res_txt = subprocess.run([command], stdout=subprocess.PIPE, shell=True).stdout.decode()
        try:
            res_txt = res_txt.split("\n")[:-1]
            current_time = float(res_txt[0])
            res.time += current_time 
            res.done = int(res_txt[1])
            res.nodes = int(res_txt[2])
            res.edges = int(res_txt[3])
        except:
            res.time = 7200
            res.done = 0
            return res
    res.time /= runs
    return res

if __name__ == "__main__":
    con = sqlite3.connect("benchlarge.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    gpu = int(argv[1])
    cur.execute('select * from results where status = 0 order by edges')
    result = cur.fetchall()

    for rc in result:
        try:
            gdata = gdata_file(rc["name"]) if rc["tool"] == "file" else gdata_law(rc["name"])
            valmari_res = Result()
            cpu_res = Result()
            cuda_res = experiment(
                CUDA_COMMAND.format(
                    inp = INPUT_FILE.format(name=rc["name"], cmd="cat") if rc["tool"] == "file" else INPUT_LAW.format(name=rc["name"], fmt="arcs-cuda"),
                    edges = 3600000000,
                    gpu = gpu
                )
            )
            cur.execute("UPDATE results SET nodes=?,edges=?,valmari=?,cpu=?,cuda=?,status=? WHERE name=?", (
                gdata.nodes,
                gdata.edges,
                json.dumps(valmari_res.__dict__),
                json.dumps(cpu_res.__dict__),
                json.dumps(cuda_res.__dict__),
                1,
                rc["name"]
            ))
            con.commit()
        except Exception as e:
            print(f"FAIL: {e}")
            cur.execute("UPDATE results SET status=? WHERE name=?", (-1, rc['name']))
            con.commit()
