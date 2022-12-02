from time import time
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

    prob = np.array(prob, dtype=np.double)
    prob /= np.sum(prob)

    np.random.choice(item, p=prob, size=100000000)

    print(time() - st)
