import os
import numpy as np
import random
import gc
from tqdm import tqdm
from time import time
from loguru import logger

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
            self.data[idA][1].append(-1)
            self.data[idA][2].append(weight)
            
            if bidirectional:
                self.data[idB][0].append(idA)
                self.data[idB][1].append(-1)
                self.data[idB][2].append(weight)
        self.i2item = np.array(self.i2item)
        return
    
    def cal_alias_table(self, info=False):
        
        to_iter = range(self.item_count)
        
        #   If debug mode (info) is on, show init process bar.
        if info:
            logger.info('building alias table')
            to_iter = tqdm(to_iter)
        
        for i in to_iter:
            #   Make data into numpy array for the best performance.
            self.data[i][0] = np.array(self.data[i][0])
            self.data[i][1] = np.array(self.data[i][1])
            self.data[i][2] = np.array(self.data[i][2])
            
            #   Start making alias table.
            mul = np.float64(len(self.data[i][0]))/np.sum(self.data[i][2])
            self.data[i][2] *= mul
            
            SA = []     # >1 stack
            SB = []     # <1 stack

            #   Sort all numbers into the stacks.
            for j in range(len(self.data[i][0])):
                if (self.data[i][2][j] - 1) >= thresh:
                    SA.append(j)
                elif (1 - self.data[i][2][j]) >= thresh:
                    SB.append(j)

            #   Determine whether to take numbers from stack or not
            SA_tak = True
            SB_tak = True
            a = None
            b = None

            #   If stack not empty, continue process.
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
                    SA_tak = False
                elif (1 - self.data[i][2][a]) >= thresh:
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

        #   If target is none, sample a random node from the graph.
        if target is None:
            randIDX = np.random.randint(self.item_count, size=size)
            returnval = self.i2item[randIDX]

        #   Else, sample a node which is connected with the target.
        else:
            target = str(target)
            targetIDX = -1
            try:
                targetIDX = self.item2i[target]
            except KeyError:
                raise KeyError("item '{item}' not in graph!".format(item=target))

            #   This special way performs well on sampling one at a time.
            if size == 1:
                randA = np.random.randint(len(self.data[targetIDX][0]))
                randB = np.random.uniform(0, 1)
                if randB > self.data[targetIDX][2][randA]:
                    returnval = self.i2item[self.data[targetIDX][1][randA]]
                else:
                    returnval = self.i2item[self.data[targetIDX][0][randA]]

            else:
                randA = np.random.randint(len(self.data[targetIDX][0]), size=size)
                thresh = self.data[targetIDX][2][randA]
                Lo = self.data[targetIDX][0][randA]
                Hi = self.data[targetIDX][1][randA]
                randB = np.random.uniform(0, 1, size=size)
                #   If random 0~1 is smaller than thresh, return item below
                #   Else return item above.
                returnval = np.where(randB <= thresh, Lo, Hi)
                returnval = self.i2item[returnval]

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
    print(s.sample("u1", size=10))
