import random
import numpy as np
from loguru import logger
from tqdm import tqdm
from sampler import sampler

SEED = 0
ALPHA_MIN = 0.0001

class LINE():
    def __init__(self, dimension: int, order: int = 2):
        np.random.seed(SEED)

        self.graph      = []
        v_num, positive_samples, itm_to_vid, vid_to_itm, usr_to_vid, vid_to_usr = self.__load_data__()

        self.v_num      = v_num
        self.dim        = int(dimension)
        self.order      = int(order)
        self.s          = sampler(self.graph, info=True)

        self.pos_lst    = positive_samples # usr to itm
        self.itm_to_vid = itm_to_vid
        self.vid_to_itm = vid_to_itm
        self.usr_to_vid = usr_to_vid
        self.vid_to_usr = vid_to_usr

        if self.order == 1:
            self.w_vertex_o1 = np.random.rand(self.v_num, self.dim)
        else:
            self.w_vertex = np.random.rand(self.v_num, self.dim)
            self.w_context = np.zeros((self.v_num, self.dim))

    def __sample_neg__(self):
        return random.randint(0, self.v_num-1)

    def __sample_src__(self):
        return random.choice(list(self.pos_lst))

    def __sample_tar__(self, v1):
        return random.choice(self.pos_lst[v1])

    def __load_data__(self):
        logger.info('loading data')

        usr_idx, itm_idx = 0, 0
        usr_to_vid = {}
        vid_to_usr = {}
        itm_to_vid = {}
        vid_to_itm = {}
        positive_samples = {}

        with open("net.dat") as file:
            line = file.readline()
            line = file.readline()

            while(line):
                usr, itm, rate = line.split(" ")

                if usr not in usr_to_vid:
                    usr_to_vid[usr] = usr_idx
                    vid_to_usr[usr_idx] = usr
                    usr_idx += 1
                if itm not in itm_to_vid:
                    itm_to_vid[itm] = itm_idx
                    vid_to_itm[itm_idx] = itm
                    itm_idx += 1
                   
                if float(rate) >= 3.0:
                    if usr_to_vid[usr] not in positive_samples:
                        positive_samples[usr_to_vid[usr]] = [itm_to_vid[itm]]
                    else:
                        positive_samples[usr_to_vid[usr]].append(itm_to_vid[itm])

                self.graph.append((usr, itm, float(rate)))

                line = file.readline()

        return usr_idx+itm_idx, positive_samples, itm_to_vid, vid_to_itm, usr_to_vid, vid_to_usr

    def train(self, sample_times: int, negative_samples: int, alpha: float):
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
        ALPHA_MIN = alpha*0.0001

        if self.order == 1:
            progress = tqdm(range(sample_times))
            for p in progress:
                # sample v1, v2
                # v1 = self.__sample_src__()
                # v2 = self.__sample_tar__(v1)
                v1_ = self.s.sample()
                if v1_ in self.usr_to_vid:
                    v1 = self.usr_to_vid[v1_]
                elif v1_ in self.itm_to_vid:
                    v1 = self.itm_to_vid[v1_]

                v2_ = self.s.sample(v1_)
                if v2_ in self.usr_to_vid:
                    v2 = self.usr_to_vid[v2_]
                elif v2_ in self.itm_to_vid:
                    v2 = self.itm_to_vid[v2_]

                # update
                ## positive
                f = np.dot(self.w_vertex_o1[v1], self.w_vertex_o1[v2])
                f = 1 / (1 + np.exp(-f))
                g = (1 - f) * alpha
                update_value = np.dot(self.w_vertex_o1[v1], g)
                total_update_value = update_value
                self.w_vertex_o1[v2] = np.add(self.w_vertex_o1[v2], update_value)
                ## negative
                for _ in range(negative_samples):
                    # v2 = self.__sample_neg__()
                    v2_ = self.s.sample()
                    if v2_ in self.usr_to_vid:
                        v2 = self.usr_to_vid[v2_]
                    elif v2_ in self.itm_to_vid:
                        v2 = self.itm_to_vid[v2_]
                    f = np.dot(self.w_vertex_o1[v1], self.w_vertex_o1[v2])
                    f = 1 / (1 + np.exp(-f))
                    g = (0 - f) * alpha
                    update_value = np.dot(self.w_vertex_o1[v1], g)
                    total_update_value =  np.add(total_update_value, update_value)
                    self.w_vertex_o1[v2] = np.add(self.w_vertex_o1[v2], update_value)
                self.w_vertex_o1[v1] = np.add(self.w_vertex_o1[v1], total_update_value)

                # loss
                alpha = alpha * (1 - (p+1)/sample_times)
                alpha = max(alpha, ALPHA_MIN)
                progress.set_description("Alpha: {Alpha:.6f}".format(Alpha=alpha))

            # save weights
            self.model = self.w_vertex_o1
        else:
            progress = tqdm(range(sample_times))
            for p in progress:
                # sample v1, v2
                # v1 = self.__sample_src__()
                # v2 = self.__sample_tar__(v1)
                v1_ = self.s.sample()
                if v1_ in self.usr_to_vid:
                    v1 = self.usr_to_vid[v1_]
                elif v1_ in self.itm_to_vid:
                    v1 = self.itm_to_vid[v1_]

                v2_ = self.s.sample(v1_)
                if v2_ in self.usr_to_vid:
                    v2 = self.usr_to_vid[v2_]
                elif v2_ in self.itm_to_vid:
                    v2 = self.itm_to_vid[v2_]

                # update
                ## positive
                f = np.dot(self.w_vertex[v1], self.w_context[v2])
                f = 1 / (1 + np.exp(-f))
                g = (1 - f) * alpha
                update_value = np.dot(self.w_vertex[v1], g)
                total_update_value = update_value
                self.w_context[v2] = np.add(self.w_context[v2], update_value)
                ## negative
                for _ in range(negative_samples):
                    # v2 = self.__sample_neg__()
                    v2_ = self.s.sample()
                    if v2_ in self.usr_to_vid:
                        v2 = self.usr_to_vid[v2_]
                    elif v2_ in self.itm_to_vid:
                        v2 = self.itm_to_vid[v2_]
                    f = np.dot(self.w_vertex[v1], self.w_context[v2])
                    f = 1 / (1 + np.exp(-f))
                    g = (0 - f) * alpha
                    update_value = np.dot(self.w_vertex[v1], g)
                    total_update_value =  np.add(total_update_value, update_value)
                    self.w_context[v2] = np.add(self.w_context[v2], update_value)
                self.w_vertex[v1] = np.add(self.w_vertex[v1], total_update_value)

                # loss
                alpha = alpha * (1 - (p+1)/sample_times)
                alpha = max(alpha, ALPHA_MIN)
                progress.set_description("Alpha: {Alpha:.6f}".format(Alpha=alpha))

            # save weights
            self.model = self.w_context

    def predict(self, topk):
        logger.info("Recommend")
        f = open("recommend.dat", "w")
        f.write("")
        f.close()

        for usr in tqdm(self.vid_to_usr):
            candidates = []
            for itm in self.vid_to_itm:
                candidates.append({
                    "item_id": self.vid_to_itm[itm],
                    "score": np.dot(self.model[usr], self.model[itm])
                })
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

            f = open("recommend.dat", "a")
            for k, c in  enumerate(candidates[:topk]):
                f.write(f"{self.vid_to_usr[usr]} {c['item_id']} {k+1}\n")
            f.close()

        
line = LINE(64, 1)
line.train(10000, 5000, 0.025)
line.predict(10)