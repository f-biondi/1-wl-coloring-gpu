import json
import subprocess
import sqlite3
import os
from sys import argv

TIMEOUT = 3600
INPUT_LAW = "./webgraph to {fmt} {name}/{name}.graph"
INPUT_FILE = "{cmd} ./datasets/{name}.txt"
CUDA_COMMAND = "{inp} | timeout {timeout} ./batched_be {edges} {gpu}" 
CPU_COMMAND = "{inp} | timeout {timeout} ./random_be" 
VALMARI_COMMAND = "{inp} | timeout {timeout} ./part_ref" 
SIGREF_COMMAND = "timeout {timeout} ./sigrefmc --workers={workers} mc-models/{name}.xctmc | ./sigref-adapter" 
TORCH_COMMAND = "{inp} | timeout {timeout} ./tbe" 

class Result():
    def __init__(self, done=0, nodes=0, edges=0, time=0):
        self.done = done
        self.nodes = nodes
        self.edges = edges
        self.time = time

class Gdata():
    def __init__(self, rc):
        #print(dict(zip(rc.keys(), rc)))
        print(rc["name"])
        self.name = rc["name"]
        self.nodes = rc["nodes"]
        self.edges = rc["edges"]
        self.tool = rc["tool"]
        self.small = bool(rc["small"])
        if self.tool == "law":
            os.system(f"./download.sh {self.name}")

    def input_command(self, fmt, cmd):
        if self.tool == "file":
            return INPUT_FILE.format(name=self.name, cmd=cmd)
        else:
            return INPUT_LAW.format(name=self.name, fmt=fmt)

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
            res.time = TIMEOUT
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
            gdata = Gdata(rc)
            # print(gdata.name)
            valmari_res = experiment(
                VALMARI_COMMAND.format(
                    inp = gdata.input_command("arcs-valmari", "./valmari-adapter"),
                    timeout = TIMEOUT,
                )
            )

            cpu_res = experiment(
                CPU_COMMAND.format(
                    inp = gdata.input_command("arcs-cuda", "cat"),
                    timeout = TIMEOUT,
                )
            )

            cuda = {}
            for p in [1,0.75,0.50,0.25]:
                cuda[p] = experiment(
                    CUDA_COMMAND.format(
                        inp = gdata.input_command("arcs-cuda", "cat"),
                        timeout = TIMEOUT,
                        edges = str(int(gdata.edges * p)),
                        gpu = gpu
                    )
                )

            sigref = {}
            for cores in [32, 64]:
                sigref[cores] = experiment(
                    SIGREF_COMMAND.format(
                        timeout = TIMEOUT,
                        workers = str(cores),
                        name = gdata.name,
                    )
                ) if gdata.small else Result()

            torch_res = experiment(
                TORCH_COMMAND.format(
                    inp = gdata.input_command("arcs-cuda", "cat"),
                    timeout = TIMEOUT,
                )
            ) if gdata.small else Result()


            #      name      │ tool │ small │  nodes   │   edges    │ valmari │ cpu │ cuda │ cuda_75 │ cuda_50 │ cuda_25 │ mc_32 │ mc_64 │ torch │ status
            cur.execute("UPDATE results SET nodes=?,edges=?,valmari=?,cpu=?,cuda=?,cuda_75=?,cuda_50=?,cuda_25=?,mc_32=?,mc_64=?,torch=?,status=? WHERE name=?", (
                gdata.nodes,
                gdata.edges,
                json.dumps(valmari_res.__dict__),
                json.dumps(cpu_res.__dict__),
                json.dumps(cuda[1].__dict__),
                json.dumps(cuda[0.75].__dict__),
                json.dumps(cuda[0.50].__dict__),
                json.dumps(cuda[0.25].__dict__),
                json.dumps(sigref[32].__dict__),
                json.dumps(sigref[64].__dict__),
                json.dumps(torch_res.__dict__),
                1,
                rc["name"]
            ))
            con.commit()
        except Exception as e:
            print(f"FAIL: {repr(e)}")
            cur.execute("UPDATE results SET status=? WHERE name=?", (-1, rc['name']))
            con.commit()

