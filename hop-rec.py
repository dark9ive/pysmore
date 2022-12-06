import argparse
import numpy as np
from loguru import logger
from tqdm import tqdm
from sampler import Sampler


SEED = 0


class HopRec():
    def __init__(self, dimension: int):
        # init random seed
        np.random.seed(SEED)

        # init parameters
        self.dimension = dimension
        self.user_embeddings = None
        self.item_embeddings = None

    def read_data(self, path: str):
        logger.info("Reading data...")
        # read user data as csv
        with open(path, 'r') as f:
            data = f.readlines()

        # 0: userId, 1: itemId, 2: rating, 3: timestamp
        logger.info("Splitting data...")
        data = [line.strip().split(',') for line in data]
        data = [(int(line[0]), int(line[1]), float(line[2]), int(line[3]))
                for line in data]
        self.num_users = max([line[0] for line in data]) + 1
        self.num_items = max([line[1] for line in data]) + 1

        ######
        self.ui_matrix = np.zeros((self.num_users, self.num_items))
        self.graph = []

        logger.info("Building user-item matrix...")
        for line in tqdm(data):
            self.ui_matrix[line[0], line[1]] = line[2]
            self.graph.append((f"u_{line[0]}", f"i_{line[1]}", 1))

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
        if u[0] != "u":
            u = self.sampler.sample(u)
        return u

    def __target_sample__(self, u: str):
        return self.sampler.sample(u)

    def __negitive_sample__(self, pos_item: str):
        neg_item = self.sampler.sample()
        if neg_item == pos_item or neg_item[0] != "i":
            neg_item = self.sampler.sample()
        return neg_item

    def updateFBPR(self, u_id: str, i_id: str, j_id: str, learning_rate: float, epsilon: float):
        """update the parameters of the model according to the feedback BPR loss

        Args:
            u (str): user id
            i (str): positive item id
            j (str): negative item id
            learning_rate (float):  learning rate
            epsilon (float): the parameter of the feedback BPR loss
            tries (int): the number of tries
        Returns:
            bool: need update or not
        """

        u = int(u_id[2:])
        i = int(i_id[2:])
        j = int(j_id[2:])

        # @ is matrix multiplication
        x_ui = self.user_embeddings[u] @ self.item_embeddings[i]
        x_uj = self.user_embeddings[u] @ self.item_embeddings[j]
        x_uij = x_ui - x_uj

        if x_uij > epsilon:
            return False

        loss = 1 / (1 + np.exp(x_uij))  # sigmoid
        self.user_embeddings[u] += learning_rate * \
            loss * (self.item_embeddings[i] - self.item_embeddings[j])
        self.item_embeddings[i] += learning_rate * \
            loss * self.user_embeddings[u]
        self.item_embeddings[j] -= learning_rate * \
            loss * self.user_embeddings[u]

    def train(self, num_epochs: int, learning_rate: float, walk_length: int):
        self.__build_graph__()
        self.__build_mf_model__()

        #
        logger.info("Training...")
        logger.info(
            f"num_epochs: {num_epochs}, learning_rate: {learning_rate}, walk_length: {walk_length}")

        for epoch in tqdm(range(num_epochs)):
            user = self.__source_sample__()
            pos_item = self.__target_sample__(user)

            margin = 1
            for step in range(2, walk_length+1):
                if step != 1:
                    pos_item = self.__target_sample__(pos_item)
                    pos_item = self.__target_sample__(pos_item)

                neg_item = self.__negitive_sample__(pos_item)
                neg_item = self.__target_sample__(user)
                self.updateFBPR(user, pos_item, neg_item,
                                learning_rate/step, margin/step)

    def predict(self, topk: int, fileName: str):
        logger.info("Predicting...")

        p = self.user_embeddings @ self.item_embeddings.T

        self.predictions = {}
        for u in range(1, self.num_users):
            self.predictions[u] = np.argsort(-p[u])
            self.predictions[u] = self.predictions[u][:topk]

        logger.info("End predicting.")

        logger.info("Saving predictions...")
        with open(fileName, 'w') as f:
            for u in range(1, self.num_users):
                f.write(
                    f"{u},{' '.join([str(i) for i in self.predictions[u]])}\n")
        logger.info("End saving predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dimension", help="MF dimension", type=int, default=10)
    parser.add_argument(
        "-e", "--num_epochs", help="number of epochs", type=int, default=10)
    parser.add_argument(
        "-l", "--learning_rate", help="learning rate", type=float, default=0.01)
    parser.add_argument(
        "-w", "--walk_length", help="walk length", type=int, default=10)
    parser.add_argument(
        "-t", "--topk", help="topk", type=int, default=10)
    args = parser.parse_args()

    fileName = f"result_d{args.dimension}_e{args.num_epochs}_l{args.learning_rate}_w{args.walk_length}_t{args.topk}.txt"

    hoprec = HopRec(dimension=args.dimension)
    hoprec.read_data(path="./data/ml-1m/ratings.csv")
    hoprec.train(num_epochs=args.num_epochs,
                 learning_rate=args.learning_rate, walk_length=args.walk_length)
    hoprec.predict(topk=args.topk, fileName=fileName)
