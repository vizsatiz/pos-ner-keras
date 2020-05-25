import uuid
from multiprocessing.pool import Pool
import nltk
import stanza
import json

from config import work_dir

tagged_sentences = nltk.corpus.treebank.tagged_sents()

print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))

# stanza.download('en')  # This downloads the English models for the neural pipeline
nlp = stanza.Pipeline('en')  # This sets up a default neural pipeline in English


def write_json(data, file_path):
    with open(file_path, "w+") as f:
        json.dump(data, f)


def check_for_word(wd, tokens):
    for tk in tokens:
        if wd in tk.text:
            return tk.type
    return "O"


sentence_tags = []


def tag_data_by_standford_stanza(ts):
    print("Starting partition size - {}".format(len(ts)))
    index = 0
    for tagged_sentence in ts:
        tags_2 = []
        sentence, tags = zip(*tagged_sentence)
        doc = nlp(" ".join(sentence))
        entities = doc.ents
        for ind, wd in enumerate(sentence):
            tags_2.append("{}:{}".format(tags[ind], check_for_word(wd, entities)))
        sentence_tags.append((sentence, tags_2))
        index += 1
    write_json(sentence_tags, "{}/{}".format(work_dir, "sentences-{}.json".format(str(uuid.uuid1()))))


p = Pool(5)
n = 500
tagged_sentences = tagged_sentences
x = list([tagged_sentences[i:i + n] for i in range(0, len(tagged_sentences), n)])
list(map(tag_data_by_standford_stanza, x))
