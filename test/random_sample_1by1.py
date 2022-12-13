from time import time
import random
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    st = time()
    item = []
    prob = []
    while True:
        try:
            itemA, itemB, weight = input().split(" ")
            item.append(itemB)
            prob.append(int(weight))
        except EOFError:
            break

    print(prob)
    prob /= np.sum(prob)
    print(item)
    print(prob)

    for i in tqdm(range(100000000)):
        random.choices(item, weights=prob, k=1)

    print(time() - st)
