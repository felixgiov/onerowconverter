import wikipedia
import nltk
import random
import math

def merge_sentences(sents1, sents2):
    output = []
    if len(sents1) == len(sents2):
        for i in range(len(sents1)):
            sent = ""
            sent = sents1[i] + "\t" + sents2[i]
            output.append(sent)

    return output


def main():
    wiki_page = wikipedia.page("World War I")
    wiki_page2 = wikipedia.page("World War II")
    corpus = wiki_page.content + ". " + wiki_page2.content

    nltk.download('punkt')
    corpus_splitted = nltk.tokenize.sent_tokenize(corpus.replace('\r', ' ').replace('\n', ' ')
                                                  .replace('===== ', '').replace('==== ', '').replace('=== ', '')
                                                  .replace('== ', '').replace('   ', ' ').replace('  ', ' '))
    print(corpus_splitted[:10])
    print(len(corpus_splitted))

    dictionary = dict.fromkeys(['1', '!', 'q', 'Q', 'a', 'A', 'z', 'Z'], "1")
    dictionary.update(dict.fromkeys(['2', '@', 'w', 'W', 's', 'S', 'x', 'X'], "2"))
    dictionary.update(dict.fromkeys(['3', '#', 'e', 'E', 'd', 'D', 'c', 'C'], "3"))
    dictionary.update(dict.fromkeys(['4', '$', 'r', 'R', 'f', 'F', 'v', 'V'], "4"))
    dictionary.update(dict.fromkeys(['5', '%', 't', 'T', 'g', 'G', 'b', 'B'], "5"))
    dictionary.update(dict.fromkeys(['6', '^', 'y', 'Y', 'h', 'H', 'n', 'N'], "6"))
    dictionary.update(dict.fromkeys(['7', '&', 'u', 'U', 'j', 'J', 'm', 'M'], "7"))
    dictionary.update(dict.fromkeys(['8', '*', 'i', 'I', 'k', 'K', ',', '<'], "8"))
    dictionary.update(dict.fromkeys(['9', '(', 'o', 'O', 'l', 'L', '.', '>'], "9"))
    dictionary.update(dict.fromkeys(['0', ')', 'p', 'P', ';', ':', '/', '?'], "0"))
    dictionary.update(dict.fromkeys(['-', '_', '[', '{', '\'', '\"'], "-"))
    dictionary.update(dict.fromkeys(['=', '+', ']', '}', '|', '\\'], "="))

    print(dictionary)

    corpus_converted = []
    for sentence in corpus_splitted:
        sentence_converted = sentence
        for char in sentence:
            if char in dictionary:
                sentence_converted = sentence_converted.replace(char, dictionary[char])
        corpus_converted.append(sentence_converted)

    print(corpus_converted[:10])
    print(len(corpus_converted))

    merged = merge_sentences(corpus_splitted, corpus_converted)

    random.seed(21)
    random.shuffle(merged)
    train_data = merged[:math.floor(0.8*len(merged))]
    eval_data = merged[math.floor(0.8*len(merged)):]

    with open("data/train.txt", "w") as writer:
        for line in train_data:
            writer.write(line+"\n")

    with open("data/eval.txt", "w") as writer:
        for line in eval_data:
            writer.write(line+"\n")


if __name__ == '__main__':
    main()

