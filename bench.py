import json
import subprocess
import sqlite3
import os
from sys import argv

CUDA_COMMAND = "{inp} | timeout 1800 ./batched_be {edges} {gpu}" 
CPU_COMMAND = "{inp} | timeout 1800 ./random_be" 
VALMARI_COMMAND = "{inp} | timeout 1800 ./part_ref" 
INPUT_LAW = "./webgraph to {fmt} {name}/{name}.graph 2>/dev/null"
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
    os.system(f"./download.sh {name} 2>/dev/null")
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
            res.time = 1800
            res.done = 0
            return res
    res.time /= runs
    return res

if __name__ == "__main__":
    con = sqlite3.connect("bench.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    gpu = int(argv[1])
    cur.execute('select * from results where status = 0 order by edges')
    result = cur.fetchall()

    for rc in result:
        try:
            gdata = gdata_file(rc["name"]) if rc["tool"] == "file" else gdata_law(rc["name"])
            valmari_res = experiment(
                VALMARI_COMMAND.format(
                    inp = INPUT_FILE.format(name=rc["name"], cmd="./vcon") if rc["tool"] == "file" else INPUT_LAW.format(name=rc["name"], fmt="arcs-valmari")
                )
            )
            cpu_res = experiment(
                CPU_COMMAND.format(
                    inp = INPUT_FILE.format(name=rc["name"], cmd="cat") if rc["tool"] == "file" else INPUT_LAW.format(name=rc["name"], fmt="arcs-cuda")
                )
            )
            cuda_res = experiment(
                CUDA_COMMAND.format(
                    inp = INPUT_FILE.format(name=rc["name"], cmd="cat") if rc["tool"] == "file" else INPUT_LAW.format(name=rc["name"], fmt="arcs-cuda"),
                    edges = str(gdata.edges),
                    gpu = gpu
                )
            )
            cuda_75_res = experiment(
                CUDA_COMMAND.format(
                    inp = INPUT_FILE.format(name=rc["name"], cmd="cat") if rc["tool"] == "file" else INPUT_LAW.format(name=rc["name"], fmt="arcs-cuda"),
                    edges = str(int(gdata.edges * 0.75)),
                    gpu = gpu
                )
            )
            cuda_50_res = experiment(
                CUDA_COMMAND.format(
                    inp = INPUT_FILE.format(name=rc["name"], cmd="cat") if rc["tool"] == "file" else INPUT_LAW.format(name=rc["name"], fmt="arcs-cuda"),
                    edges = str(int(gdata.edges * 0.50)),
                    gpu = gpu
                )
            )
            cuda_25_res = experiment(
                CUDA_COMMAND.format(
                    inp = INPUT_FILE.format(name=rc["name"], cmd="cat") if rc["tool"] == "file" else INPUT_LAW.format(name=rc["name"], fmt="arcs-cuda"),
                    edges = str(int(gdata.edges * 0.25)),
                    gpu = gpu
                )
            )
            #│      name      │ tool │ nodes │ edges │ valmari │ cpu │ cuda │ cuda_75 │ cuda_50 │ cuda_25 │ status │
            cur.execute("UPDATE results SET nodes=?,edges=?,valmari=?,cpu=?,cuda=?,cuda_75=?,cuda_50=?,cuda_25=?,status=? WHERE name=?", (
                gdata.nodes,
                gdata.edges,
                json.dumps(valmari_res.__dict__),
                json.dumps(cpu_res.__dict__),
                json.dumps(cuda_res.__dict__),
                json.dumps(cuda_75_res.__dict__),
                json.dumps(cuda_50_res.__dict__),
                json.dumps(cuda_25_res.__dict__),
                1,
                rc["name"]
            ))
            con.commit()
        except Exception as e:
            print(f"FAIL: {e}")
            cur.execute("UPDATE results SET status=? WHERE name=?", (-1, rc['name']))
            con.commit()

