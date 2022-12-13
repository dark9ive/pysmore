import os
import numpy as np
import random
import gc
from tqdm import tqdm
from time import time
from loguru import logger
import multiprocessing

thresh = 0.000001

class sampler:
    def init_dict(self, graph, bidirectional=True, info=False):
        self.item2i = {}
        self.i2item = []
        self.data = []
        self.item_count = 0
        to_iter = graph
        if info:
            logger.info('building dictionary')
            to_iter = tqdm(to_iter)
        for itemA, itemB, weight in to_iter:
            itemA = str(itemA)
            itemB = str(itemB)
            weight = np.float64(weight)
            if itemA not in self.item2i:
                self.item2i[itemA] = self.item_count
                self.i2item.append(itemA)
                self.data.append([[], [], []])
                self.item_count += 1
            if itemB not in self.item2i:
                self.item2i[itemB] = self.item_count
                self.i2item.append(itemB)
                self.data.append([[], [], []])
                self.item_count += 1
            idA = self.item2i[itemA]
            idB = self.item2i[itemB]
            
            self.data[idA][0].append(idB)
            self.data[idA][1].append(None)
            self.data[idA][2].append(weight)
            
            if bidirectional:
                self.data[idB][0].append(idA)
                self.data[idB][1].append(None)
                self.data[idB][2].append(weight)
                #raise ValueError("duplicated edge of item '{A}' and '{B}'.".format(A=itemA, B=itemB))
        self.i2item = np.array(self.i2item)
        return
    
    def cal_alias_table(self, info=False):
        to_iter = range(self.item_count)
        if info:
            logger.info('building alias table')
            to_iter = tqdm(to_iter)
        for i in to_iter:
            self.data[i][0] = np.array(self.data[i][0])
            self.data[i][1] = np.array(self.data[i][1])
            self.data[i][2] = np.array(self.data[i][2])
            
            mul = np.float64(len(self.data[i][0]))/np.sum(self.data[i][2])
            self.data[i][2] *= mul
            
            SA = []     # >1 stack
            SB = []     # <1 stack

            for j in range(len(self.data[i][0])):
                if (self.data[i][2][j] - 1) >= thresh:
                    SA.append(j)
                elif (1 - self.data[i][2][j]) >= thresh:
                    SB.append(j)
            SA_tak = True
            SB_tak = True
            a = None
            b = None
            while SA or SB:
                if SA_tak:
                    a = SA.pop()
                if SB_tak:
                    b = SB.pop()
                SA_tak = True
                SB_tak = True

                self.data[i][1][b] = self.data[i][0][a]
                self.data[i][2][a] -= 1 - self.data[i][2][b]
                if (self.data[i][2][a] - 1) >= thresh:
                    #SA.append(a)
                    SA_tak = False
                elif (1 - self.data[i][2][a]) >= thresh:
                    #SB.append(a)
                    b = a
                    SB_tak = False
                else:
                    self.data[i][2][a] = np.float64(1)

    def __init__(self, graph, bidirectional=True, info=False):
        '''
            graph is a array of tuple (itemA, itemB, weight)
        '''
        self.init_dict(graph, bidirectional, info=info)
        self.cal_alias_table(info=info)
        gc.collect()
        return

    def sample(self, target=None, size=1):
        returnval = None
        if target is None:
            randIDX = np.random.randint(self.item_count, size=size)
            returnval = self.i2item[randIDX]
        else:
            target = str(target)
            targetIDX = -1
            try:
                targetIDX = self.item2i[target]
            except KeyError:
                raise KeyError("item '{item}' not in graph!".format(item=target))
            randA = np.random.randint(len(self.data[targetIDX][0]), size=size)
            thresh = self.data[targetIDX][2][randA]
            Hi = self.data[targetIDX][1][randA]
            Lo = self.data[targetIDX][0][randA]
            randB = np.random.uniform(0, 1)
            returnval = np.where(randB <= thresh, Hi, Lo)
            '''
            if randB > self.data[targetIDX][2][randA]:
                returnval = self.i2item[self.data[targetIDX][1][randA]]
            else:
                returnval = self.i2item[self.data[targetIDX][0][randA]]
            '''
        return returnval


if __name__ == '__main__':
    data = []
    while True:
        try:
            itemA, itemB, weight = input().split(" ")
            data.append((itemA, itemB, weight))
        except EOFError:
            break
    s = sampler(data, info=True)
    print(s.sample(size=10))
