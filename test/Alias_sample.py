from time import time
from sampler import sampler
from tqdm import tqdm
from loguru import logger

if __name__ == '__main__':
    st = time()
    data = []
    while True:
        try:
            itemA, itemB, weight = input().split(" ")
            data.append((itemA, itemB, weight))
        except EOFError:
            break

    s = sampler(data, info=True)
    logger.info("sampling")

    for i in tqdm(range(100000000)):
        s.sample("u1")

    print(time() - st)
