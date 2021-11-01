import codecs
import copy
import random
import math
import time


import numpy as np
from numpy import linalg as la
from copyf import deepcopy as dc

entity_dic = {}  # dictionary to store each entity with its index as value
relation_dic = {}  # dictionary to store each relation with its index as value
triple_list = []  # list to store training set using the indexes
entity_set = set()  # set to store all entity indexes (random order)
relation_set = set()  # set to store all relation indexes (random order)


# load the FB15k data into entity relation as dictionary: {'mid code': index}
# entire relation as triple list: [head(index), tail(index), relation(index)]
def load_data(path):
    entity_file_path = path + "\\entity2id.txt"
    relation_file_path = path + "\\relation2id.txt"
    train_file_path = path + "\\train.txt"

    with open(entity_file_path, "r") as e, \
            open(relation_file_path, "r") as r, \
            open(train_file_path, "r") as t:

        for line in e.readlines():
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            entity_dic[line[0]] = line[1]

        for line in r.readlines():
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            relation_dic[line[0]] = line[1]

        # triple_list contains indexes of entity and relation in training set
        triple_list.clear()  # clear the list to prevent duplicates
        for line in t.readlines():
            line = line.strip().split('\t')
            if len(line) < 3:
                continue
            triple_list.append([entity_dic[line[0]], entity_dic[line[1]], relation_dic[line[2]]])
            entity_set.add(entity_dic[line[0]])
            entity_set.add(entity_dic[line[1]])
            relation_set.add(relation_dic[line[2]])

        print("Load complete")


def distanceL1(head, rel, tail):
    return np.sum(np.fabs(head + rel - tail))


def distanceL2(head, rel, tail):
    h = np.array(head)
    r = np.array(rel)
    t = np.array(tail)
    s = h + r - t
    return np.linalg.norm(s)


class TransE:
    def __init__(self, triple=None, triple_c=None, entity=None,
                 relation=None, dim=100, lr=0.01, margin=1, loss=0, L1=True):
        if triple is None:
            triple = triple_list
        if triple_c is None:
            triple_c = triple_list
        if entity is None:
            entity = entity_set
        if relation is None:
            relation = relation_set
        self.triple = triple
        self.triple_c = triple_c  # create a new field for corrupted triplet to prevent repeating sampling
        self.entity = entity  # will be a dict after initialization with value being the embedding vector
        self.relation = relation  # will be a dict after initialization
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.loss = loss
        self.L1 = L1

    # initialize the TransE model with uniformly random entity and relation embeddings
    def initialize(self):
        entity_unif = {}
        relation_unif = {}
        k = self.dim

        for entity in self.entity:
            e_temp = np.random.uniform(-6 / math.sqrt(k), 6 / math.sqrt(k), k)
            entity_unif[entity] = e_temp / la.norm(e_temp)  # normalize e

        for relation in self.relation:
            l_temp = np.random.uniform(-6 / math.sqrt(k), 6 / math.sqrt(k), k)
            relation_unif[relation] = l_temp / la.norm(l_temp)  # normalize l

        self.entity = entity_unif
        self.relation = relation_unif
        print("Initialization completed")

    def sample_s_batch(self, size) -> list:
        return random.sample(self.triple, size)

    # define the triple_c field of transE model: take random item in
    # entity set to replace the corresponding head xor tail
    def corrupt_triple(self):
        triple_c = dc(self.triple)
        for triplet in triple_c:
            k = random.random()
            temp0 = triplet[0]
            temp1 = triplet[1]
            if k < 0.5:
                while True:
                    triplet[0] = random.randint(0, len(entity_set) - 1)
                    if triplet[0] != temp0:
                        break
            else:
                while True:
                    triplet[1] = random.randint(0, len(entity_set) - 1)
                    if triplet[1] != temp1:
                        break
        self.triple_c = triple_c

    def train(self, batches: int, epochs: int):
        batch_size = len(self.triple) // batches
        for epoch in range(epochs):
            start = time.time()
            print(f"epoch: {epoch}")
            self.loss = 0
            for batch in range(batches):
                # print(f"batch: {batch}")
                s_batch = []
                t_batch = []
                rand_list = random.sample(range(len(self.triple)), batch_size)
                # create s_batch and t_batch
                for index in rand_list:
                    to_add = (self.triple[index], self.triple_c[index])
                    s_batch.append(to_add[0])
                    t_batch.append(to_add)
                self.update(t_batch)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)

            if epoch % 20 == 0:
                with codecs.open("entity_temp_trained", "w") as en_temp:
                    for e in self.entity.keys():
                        en_temp.write(e + "\t")
                        en_temp.write(str(list(self.entity[e])))
                        en_temp.write("\n")
                with codecs.open("relation_temp_trained", "w") as re_temp:
                    for l in self.relation.keys():
                        re_temp.write(l + "\t")
                        re_temp.write(str(list(self.relation[l])))
                        re_temp.write("\n")
        print("writing result...")
        with codecs.open("entity_trained", "w") as en:
            for e in self.entity.keys():
                en.write(e + "\t")
                en.write(str(list(self.entity[e])))
                en.write("\n")
        with codecs.open("relation_trained", "w") as re:
            for l in self.relation.keys():
                re.write(l + "\t")
                re.write(str(list(self.relation[l])))
                re.write("\n")
        print("training completed")

    def update(self, batch: list):
        entity_result = dc(self.entity)
        relation_result = dc(self.relation)
        # batch = [correct triple, corrupt triple] (indexes)
        #       = [(h, t, l), (h', t', l)]
        for triplet, triplet_c in batch:

            h = self.entity[str(triplet[0])]
            t = self.entity[str(triplet[1])]
            l = self.relation[str(triplet[2])]

            h_prime = self.entity[str(triplet_c[0])]
            t_prime = self.entity[str(triplet_c[1])]

            # compute d(h + l, t) and d(h' + l, t')
            if self.L1:
                dist = distanceL1(h, l, t)
                dist_prime = distanceL1(h_prime, l, t_prime)
            else:
                dist = distanceL2(h, l, t)
                dist_prime = distanceL2(h_prime, l, t_prime)

            # compute [\gamma + d(h + l, t) - d(h' + l, t')]_+
            err = max(0, self.margin + dist - dist_prime)

            # update w.r.t gradient
            if err > 0:
                self.loss += err

                # for L2 distance
                grad_pos = 2 * (h + l - t)
                grad_neg = 2 * (h_prime + l - t_prime)

                # for L1 distance
                if self.L1:
                    grad_pos = np.array([1 if g >= 0 else -1 for g in grad_pos])
                    grad_neg = np.array([1 if g >= 0 else -1 for g in grad_neg])

                pos_update = self.lr * grad_pos
                neg_update = self.lr * grad_neg

                entity_result[str(triplet[0])] -= pos_update
                entity_result[str(triplet[1])] += pos_update

                if triplet[0] == triplet_c[0]:  # tail is corrupted
                    entity_result[str(triplet[0])] += neg_update
                    entity_result[str(triplet_c[1])] -= neg_update
                elif triplet[1] == triplet_c[1]:  # head is corrupted
                    entity_result[str(triplet_c[0])] += neg_update
                    entity_result[str(triplet[1])] -= neg_update
                relation_result[str(triplet[2])] -= pos_update
                relation_result[str(triplet[2])] += neg_update
        for key in entity_result.keys():
            entity_result[key] /= la.norm(entity_result[key])

        for key in relation_result.keys():
            relation_result[key] /= la.norm(relation_result[key])

        self.entity = entity_result
        self.relation = relation_result


def main():
    print("load file...")
    load_data("FB15k")
    print(f"Complete load. entity : {len(entity_set)} ,\
         relation : {len(relation_set)} ,\
          triple : {len(triple_list)}")
    model = TransE(dim=50, lr=0.01, margin=1, L1=False)
    model.initialize()
    model.corrupt_triple()
    model.train(batches=400, epochs=200)


if __name__ == '__main__':
    main()
