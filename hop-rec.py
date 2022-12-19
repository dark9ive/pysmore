import argparse
import numpy as np
from loguru import logger
from tqdm import tqdm
from sampler import Sampler

SEED = 0
TIMES_PER_SAMPLE = 1000


class HopRec():
    def __init__(self, dimension: int):
        # init random seed
        np.random.seed(SEED)

        # init parameters
        self.dimension = dimension
        self.user_embeddings = None
        self.item_embeddings = None

        self.user_index = {}
        self.item_index = {}

        self.num_users = 0
        self.num_items = 0

    def read_data(self, path: str, field_path: str):
        logger.info("Reading field...")
        # build field info
        with open(field_path, "r") as f:
            lines = f.readlines()
        lines = [line.strip().split(" ") for line in lines]
        field = {line[0]: line[1] for line in lines}

        logger.info("Reading data...")
        # read user data
        with open(path, 'r') as f:
            data = f.readlines()
        # 0: nodeA, 1: nodeB, 2: weight
        logger.info("Splitting data...")
        data = [line.strip().split(" ") for line in data]
        data = [(line[0], line[1], float(line[2])) for line in data]

        # # get user and item index
        logger.info("Building user and item index...")
        for line in tqdm(data):
            for i in range(2):
                node = line[i]
                assert node in field
                vertex_field = field[node]
                if vertex_field == "u":
                    if node not in self.user_index:
                        self.user_index[node] = self.num_users
                        self.num_users += 1
                elif vertex_field == "i":
                    if node not in self.item_index:
                        self.item_index[node] = self.num_items
                        self.num_items += 1

        ######
        self.ui_matrix = np.zeros((self.num_users, self.num_items))
        self.graph = []

        logger.info("Building user-item matrix...")
        for line in tqdm(data):
            if field[line[0]] == "u" and field[line[1]] == "i":
                u_id = line[0]
                i_id = line[1]
                u_index = self.user_index[line[0]]
                i_index = self.item_index[line[1]]
            elif field[line[1]] == "u" and field[line[0]] == "i":
                u_id = line[1]
                i_id = line[0]
                u_index = self.user_index[line[1]]
                i_index = self.item_index[line[0]]
            self.ui_matrix[u_index, i_index] = line[2]
            self.graph.append((u_id, i_id, 1))

        logger.info("End reading data.")

    def __build_graph__(self):
        logger.info("Building graph...")

        self.sampler = Sampler(self.graph, bidirectional=True, info=True)
        logger.info("End building graph.")

    def __build_mf_model__(self):
        logger.info("Building MF model...")
        self.user_embeddings = np.random.normal(
            size=(self.num_users, self.dimension))
        self.item_embeddings = np.random.normal(
            size=(self.num_items, self.dimension))
        logger.info("End building MF model.")

    def __source_sample__(self):
        u = self.sampler.sample()
        while u[0] != "u":
            u = self.sampler.sample(u)
        return u

    def __target_sample__(self, u: str):
        return self.sampler.sample(u)

    def __negitive_sample__(self, pos_item: str):
        neg_item = self.sampler.sample()
        while neg_item == pos_item or neg_item[0] != "i":
            neg_item = self.sampler.sample()
        return neg_item

    def updateFBPRPair(self, u_id: str, i_id: str, learning_rate: float, epsilon: float, lambda_: float, negative_sample_times: int):
        up = 0

        user_loss = np.zeros(self.dimension)
        for w_ in range(negative_sample_times):
            u = self.user_index[u_id]
            i = self.item_index[i_id]
            j = self.item_index[self.__negitive_sample__(i_id)]

            # @ is matrix multiplication
            x_ui = self.user_embeddings[u] @ self.item_embeddings[i]
            x_uj = self.user_embeddings[u] @ self.item_embeddings[j]
            x_uij = x_ui - x_uj

            # if x_uij > epsilon:
            # continue
            up += 1
            loss = 1 / (1 + np.exp(-x_uij))  # sigmoid

            user_loss += loss * learning_rate * \
                (self.item_embeddings[i] - self.item_embeddings[j])

            # fix the item embeddings
            self.item_embeddings[i] -= learning_rate * \
                lambda_ * self.item_embeddings[i]
            self.item_embeddings[j] -= learning_rate * \
                lambda_ * self.item_embeddings[j]

            # update
            self.item_embeddings[i] += loss * \
                learning_rate * self.user_embeddings[u]
            self.item_embeddings[j] -= loss * \
                learning_rate * self.user_embeddings[u]

        if up > 0:
            self.user_embeddings[u] -= learning_rate * \
                lambda_ * 10 * self.user_embeddings[u]
            self.user_embeddings[u] += user_loss / up

    def train(self, sample_times: int, learning_rate: float, walk_steps: int, lambda_: float, negative_sample_times: int, save_path: str):
        self.__build_graph__()
        self.__build_mf_model__()

        # training info
        logger.info("Training...")
        logger.info(
            f"sample_time: {sample_times}, learning_rate: {learning_rate}, walk_steps: {walk_steps}, lambda: {lambda_}, negative_sample_times: {negative_sample_times}")
        info = "Model Setting:\n"
        info += f"\tdimension: {self.dimension}\n"
        info += f"Model:\n"
        info += f"\tHOP-REC(HBPR)\n"
        info += f"Learning Parameters:\n"
        info += f"\tsample_times: {sample_times}\n"
        info += f"\tlearning_rate: {learning_rate}\n"
        info += f"\twalk_steps: {walk_steps}\n"
        info += f"\tlambda: {lambda_}\n"
        info += f"\tnegative_sample_times: {negative_sample_times}\n"
        print(info)

        learning_rate_origin = learning_rate
        # start training
        num_epochs = sample_times * TIMES_PER_SAMPLE

        for epoch in tqdm(range(num_epochs)):
            user = self.__source_sample__()
            pos_item = self.__target_sample__(user)

            margin = 1
            for step in range(1, walk_steps+1):
                if step != 1:
                    pos_item = self.__target_sample__(pos_item)
                    pos_item = self.__target_sample__(pos_item)

                self.updateFBPRPair(user, pos_item,
                                    learning_rate/step, margin/step, lambda_, negative_sample_times)
            # update learning rate
            learning_rate = learning_rate_origin * (1 - epoch / num_epochs)
        logger.info("End training.")
        logger.info("Saving model...")
        index_to_user = {v: k for k, v in self.user_index.items()}
        index_to_item = {v: k for k, v in self.item_index.items()}
        f = open(save_path, 'w')
        f.write(f"{self.num_users+self.num_items} {self.dimension}\n")
        for u in range(0, self.num_users):
            f.write(
                f"{index_to_user[u]} {' '.join([str(i) for i in self.user_embeddings[u]])}\n")
        for i in range(0, self.num_items):
            f.write(
                f"{index_to_item[i]} {' '.join([str(i) for i in self.item_embeddings[i]])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dimension", help="MF dimension", type=int, default=64)
    parser.add_argument(
        "-s", "--sample_times", help="sample times", type=int, default=10)
    parser.add_argument(
        "-lr", "--learning_rate", help="learning rate", type=float, default=0.025)
    parser.add_argument(
        "-w", "--walk_steps", help="walk steps", type=int, default=40)
    parser.add_argument(
        "-t", "--topk", help="topk", type=int, default=10)
    parser.add_argument(
        "-l", "--lambda", help="lambda", type=float, default=0.0025, dest="lambda_")
    parser.add_argument(
        "-n", "--negative", help="negative", type=int, default=5)
    parser.add_argument(
        "-i", "--input", help="input path", type=str)
    parser.add_argument(
        "-f", "--field", help="hop-rec field", type=str)
    parser.add_argument(
        "-o", "--output", help="output path", type=str, default="result")

    args = parser.parse_args()

    DIMENSION = args.dimension
    SAMPLE_TIMES = args.sample_times
    LEARNING_RATE = args.learning_rate
    WALK_STEPS = args.walk_steps
    TOPK = args.topk
    LAMBDA = args.lambda_
    NEGATIVE = args.negative
    INPUT_PATH = args.input
    FIELD = args.field
    OUTPUT_PATH = args.output

    # fileName = f"result_d{DIMENSION}_s{SAMPLE_TIMES}_lr{str(LEARNING_RATE)[2:]}_w{WALK_STEPS}_l{LAMBDA}_n{NEGATIVE}.txt"

    hoprec = HopRec(dimension=DIMENSION)
    hoprec.read_data(path=INPUT_PATH, field_path=FIELD)
    hoprec.train(sample_times=SAMPLE_TIMES,
                 learning_rate=LEARNING_RATE,
                 walk_steps=WALK_STEPS,
                 lambda_=LAMBDA,
                 negative_sample_times=NEGATIVE,
                 save_path=OUTPUT_PATH)
