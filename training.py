from keras import backend as K

from config import work_dir
from helpers import get_all_file_names_in_dir, get_json_from_json_file
from model import Network


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy


def read_data():
    files = get_all_file_names_in_dir(work_dir, ".json")
    sentences, sentence_tags = [], []
    for js in files:
        json_data = get_json_from_json_file("{}/{}".format(work_dir, js))
        for jn in json_data:
            sentences.append(jn[0])
            sentence_tags.append(jn[1])

    ner_tags = ["MONEY", "ORG", "PERSON", "PRODUCT", "LOC", "DATE", "TIME", "GPE"]
    clean_tags = []
    for tg in sentence_tags:
        sn_tags = []
        for sn_tag in tg:
            sp = sn_tag.split(":")
            if len(sp) == 2:
                if sp[1] in ner_tags:
                    sn_tags.append(sp[1])
                else:
                    sn_tags.append(sp[0])
            else:
                sn_tags.append(sp[0])
        clean_tags.append(sn_tags)
    return sentences, clean_tags


def start_training():
    sentences, clean_tags = read_data()
    ner_model = Network(sentences, clean_tags)
    ner_model.init(ignore_class_accuracy)
    ner_model.train()


if __name__ == "__main__":
    start_training()
