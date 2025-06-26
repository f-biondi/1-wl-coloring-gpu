with open("graph.txt", "r") as f:
    ns = list(map(lambda x: x-1, map(int," ".join(f.read().split("\n")).split(" ")[5:-1])))
    for i in range(0, len(ns),3):
        print(ns[i],ns[i+2])
