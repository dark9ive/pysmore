import argparse
from tqdm import tqdm

RATING_PATH = ""
TOP_K = 10
PREDICT_PATH = ""


def score():
    rating_file = open(RATING_PATH, encoding="utf-8", errors='ignore')
    rating_lines = rating_file.readlines()
    rating_file.close()

    predict_file = open(PREDICT_PATH, encoding="utf-8", errors='ignore')
    predict_lines = predict_file.readlines()
    predict_file.close()

    ratings = {}
    predicts = {}

    for rate in tqdm(rating_lines):
        usr, itm, _, __ = rate.split(",")

        if usr not in ratings:
            ratings[usr] = [itm]
        else:
            ratings[usr].append(itm)

    for predict in tqdm(predict_lines):
        usr, itm, = predict.split(",")

        if usr not in predicts:
            predicts[usr] = []
            itm = itm.split(" ")
            for i in itm:
                predicts[usr].append(i)

    recall = recall_at_k(ratings, predicts)
    m_ap = map_at_k(ratings, predicts)

    print("Map @ {} = {}".format(TOP_K, m_ap, TOP_K))
    print("Recall @ {} = {}".format(TOP_K, recall, TOP_K))
    # print("Precision @ {} = {}".format(k, precision_at_k))


def map_at_k(actuals, predictions, k=TOP_K):
    m_ap = 0

    for a_key, p_key in zip(actuals, predictions):
        ap, hit = 0, 0
        for k in range(TOP_K):
            if predictions[p_key][k] in actuals[a_key]:
                hit += 1
                ap += hit/(k+1)
        if hit:
            ap /= hit
        m_ap += ap

    m_ap /= len(actuals)
    return m_ap


def recall_at_k(actuals, predictions, k=TOP_K):
    avg_recall = 0

    for a_key, p_key in zip(actuals, predictions):
        truth_set = set(actuals[a_key])
        pred_set = set(predictions[p_key][:k])
        result = round(len(truth_set & pred_set) / float(len(truth_set)), 2)
        avg_recall += result

    avg_recall /= len(actuals)
    return avg_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-path", "--path",
                        help="Specify dataset path", default="net.dat")
    parser.add_argument(
        "-k", "--k", help="Set map at top k elements", type=int, default=10)
    parser.add_argument(
        "-pre", "--pre", help="Set predicted file path", default="predict.dat")

    args = parser.parse_args()

    RATING_PATH = args.path
    TOP_K = args.k
    PREDICT_PATH = args.pre

    score()
