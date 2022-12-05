import argparse
import numpy as np
from tqdm import tqdm

TOP_K = 0
GRAPH_PATH = ""
MODEL_PATH = ""
PREDICT_PATH = ""

def predict():
    graph_file = open(GRAPH_PATH, encoding="utf-8", errors='ignore')
    graph_lines = graph_file.readlines()
    graph_file.close()

    model_file = open(MODEL_PATH, encoding="utf-8", errors='ignore')
    model_lines = model_file.readlines()
    model_file.close()

    usr_idx, itm_idx = 0, 0
    usr_to_idx = {}
    idx_to_usr = {}
    itm_to_idx = {}
    idx_to_itm = {}

    for edge in tqdm(graph_lines):
        usr, itm, _ = edge.split(" ")

        if usr not in usr_to_idx:
            usr_to_idx[usr] = usr_idx
            idx_to_usr[usr_idx] = usr
            usr_idx += 1
        if itm not in itm_to_idx:
            itm_to_idx[itm] = itm_idx
            idx_to_itm[itm_idx] = itm
            itm_idx += 1

    usrs_embedding = [[] for _ in range(usr_idx)]
    itms_embedding = [[] for _ in range(itm_idx)]

    for line in tqdm(model_lines[1:]):
        line = line.split(" ")
        vertex = line[0]
        embedding = [ float(x) for x in line[1:] ]
        
        if vertex in usr_to_idx:
            usrs_embedding[usr_to_idx[vertex]] = embedding
        if vertex in itm_to_idx:
            itms_embedding[itm_to_idx[vertex]] = embedding

    usrs_embedding = np.array(usrs_embedding)
    itms_embedding = np.array(itms_embedding)

    predict_file = open(PREDICT_PATH, "w")
    predict_file.write("")
    predict_file.close()

    for usr in tqdm(usr_to_idx):
        candidates = []
        for itm in itm_to_idx:
            candidates.append({
                "itm": itm,
                "score": np.dot(usrs_embedding[usr_to_idx[usr]], itms_embedding[itm_to_idx[itm]])
            })
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        predict_file = open(PREDICT_PATH, "a")
        for k, candidate in  enumerate(candidates[:TOP_K]):
            predict_file.write(f"{usr} {candidate['itm']} {k+1}\n")
        predict_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-graph", "--graph", help="Specify graph path", default="net.dat")
    parser.add_argument("-model", "--model", help="Specify model path", default="rep_line2.txt")
    parser.add_argument("-pre", "--pre", help="Set predicted file path", default="predict.dat")
    parser.add_argument("-k", "--k", help="Set map at top k elements", type=int, default=10)


    args = parser.parse_args()

    GRAPH_PATH = args.graph
    MODEL_PATH = args.model
    PREDICT_PATH = args.pre
    TOP_K = args.k

    predict()