def generate_seq(k,length,seed):
    import random; random.seed(seed)
    temp=[tuple(sorted(random.sample(range(k),2))+[random.randint(5,10)]) for _ in range(length)]
    graph=[]
    edge_sets=set()
    for (u,v,l) in temp:
        if (u,v) not in edge_sets and (v,u) not in edge_sets:
            graph.append((u,v,l))
            edge_sets.add((u,v))
    return graph




