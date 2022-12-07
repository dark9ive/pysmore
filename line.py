import argparse
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from sampler import sampler

class line():
    def __init__(self, order, dimension, net_path):
        np.random.seed(0)

        self.vtx_to_id = {}
        self.id_to_vtx = {}

        self.sampler = None
        self.v_num = self.__load_data__(net_path)
        self.dim   = int(dimension)
        self.order = int(order)

        if self.order == 1:
            self.vertex_o1 = (np.random.rand(self.v_num, self.dim) - 0.5) / self.dim
        elif self.order == 2:
            self.w_vertex  = (np.random.rand(self.v_num, self.dim) - 0.5) / self.dim
            self.w_context = np.zeros((self.v_num, self.dim))

    def __load_data__(self, net_path):
        graph = []
        v_num = 0

        with open(net_path) as file:
            line = file.readline()
            while(line):
                usr, itm, weight = line.split(" ")

                if usr not in self.vtx_to_id:
                    self.vtx_to_id[usr] = v_num
                    self.id_to_vtx[v_num] = usr
                    v_num += 1
                if itm not in self.vtx_to_id:
                    self.vtx_to_id[itm] = v_num
                    self.id_to_vtx[v_num] = itm
                    v_num += 1

                graph.append((usr, itm, float(weight)))
                line = file.readline()

        self.sampler = sampler(graph, info=True)
        return v_num

    def train(self, sample_times, negative_samples, alpha):
        logger.info('''
            Model: 
                [LINE]
            
            Order:        {}
            Dimension:    {}
            Total Vertex: {}
            
            Learning Parameters:
                sample_times:       {}
                negative_samples:   {}
                alpha:              {}
        '''.format(self.order, self.dim, self.v_num, sample_times, negative_samples, alpha))

        sample_times = sample_times * 50000
        alpha_cur = alpha
        alpha_min = alpha * 0.0001    

        if self.order == 1:
            progress = tqdm(range(sample_times))
            for p in progress:
                # sample
                v1 = self.sampler.sample()
                v2 = self.sampler.sample(v1)
                v1_id = self.vtx_to_id[v1]
                v2_id = self.vtx_to_id[v2]

                # update
                ## positive
                f = np.dot(self.vertex_o1[v1_id], self.vertex_o1[v2_id])
                f = 1 / (1 + np.exp(-f))
                g = (1 - f) * alpha_cur
                v1_loss = np.dot(self.vertex_o1[v2_id], g)
                v2_loss = np.dot(self.vertex_o1[v1_id], g)
                self.vertex_o1[v2_id] = np.add(self.vertex_o1[v2_id], v2_loss)
                ## negative
                for _ in range(negative_samples):
                    v2 = self.sampler.sample()
                    v2_id = self.vtx_to_id[v2]

                    f = np.dot(self.vertex_o1[v1_id], self.vertex_o1[v2_id])
                    f = 1 / (1 + np.exp(-f))
                    g = (0 - f) * alpha_cur
                    v1_add_loss = np.dot(self.vertex_o1[v2_id], g)
                    v1_loss = np.add(v1_loss, v1_add_loss)
                    v2_loss = np.dot(self.vertex_o1[v1_id], g)
                    self.vertex_o1[v2_id] = np.add(self.vertex_o1[v2_id], v2_loss)
                self.vertex_o1[v1_id] = np.add(self.vertex_o1[v1_id], v1_loss)
                    
                # loss
                alpha_cur = alpha * (1 - (p)/sample_times)
                alpha_cur = max(alpha_cur, alpha_min)
                progress.set_description("Alpha: {Alpha:.8f}".format(Alpha=alpha_cur))

        elif self.order == 2:
            progress = tqdm(range(sample_times))
            for p in progress:
                # sample
                v1 = self.sampler.sample()
                v2 = self.sampler.sample(v1)
                v1_id = self.vtx_to_id[v1]
                v2_id = self.vtx_to_id[v2]

                # update
                ## positive
                f = np.dot(self.w_vertex[v1_id], self.w_context[v2_id])
                f = 1 / (1 + np.exp(-f))
                g = (1 - f) * alpha_cur
                v1_loss = np.dot(self.w_context[v2_id], g)
                v2_loss = np.dot(self.w_vertex[v1_id], g)
                self.w_context[v2_id] = np.add(self.w_context[v2_id], v2_loss)
                ## negative
                for _ in range(negative_samples):
                    v2 = self.sampler.sample()
                    v2_id = self.vtx_to_id[v2]

                    f = np.dot(self.w_vertex[v1_id], self.w_context[v2_id])
                    f = 1 / (1 + np.exp(-f))
                    g = (0 - f) * alpha_cur
                    v1_add_loss = np.dot(self.w_context[v2_id], g)
                    v1_loss = np.add(v1_loss, v1_add_loss)
                    v2_loss = np.dot(self.w_vertex[v1_id], g)
                    self.w_context[v2_id] = np.add(self.w_context[v2_id], v2_loss)
                self.w_vertex[v1_id] = np.add(self.w_vertex[v1_id], v1_loss)
                    
                # loss
                alpha_cur = alpha * (1 - (p)/sample_times)
                alpha_cur = max(alpha_cur, alpha_min)
                progress.set_description("Alpha: {Alpha:.8f}".format(Alpha=alpha_cur))


    def save(self, path):
        with open(path, "w") as f:
            f.write(f"{self.v_num} {self.dim}\n")
            if self.order == 1:
                for i in tqdm(range(len(self.vertex_o1))):
                    f.write(f"{self.id_to_vtx[i]}")
                    for elm in self.vertex_o1[i]:
                        f.write(f" {elm}")
                    f.write("\n")
            elif self.order == 2:
                for i in tqdm(range(len(self.w_vertex))):
                    f.write(f"{self.id_to_vtx[i]}")
                    for elm in self.w_vertex[i]:
                        f.write(f" {elm}")
                    f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-order", "--order", help="Set 1 for first order, 2 for second order", type=int, default=2)
    parser.add_argument("-dimensions", "--dimensions", help="Set dimensions", type=int, default=64)
    parser.add_argument("-save", "--save", help="Set save model file path", default="model_line.txt")
    parser.add_argument("-sample_times", "--sample_times", help="Set sample times", type=int, default=10)
    parser.add_argument("-negative_samples", "--negative_samples", help="Set negative samples", type=int, default=5)
    parser.add_argument("-alpha", "--alpha", help="Set alpha", type=float, default=0.025)
    parser.add_argument("-net", "--net", help="Set graph file path", default="net.dat")

    args = parser.parse_args()

    line = line(args.order, args.dimensions, args.net)
    line.train(args.sample_times, args.negative_samples, args.alpha)
    line.save(args.save)