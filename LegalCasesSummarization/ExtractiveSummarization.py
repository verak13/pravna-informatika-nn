import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
import os


def read_article(text):
    return sent_tokenize(text)


# Create vectors and calculate cosine similarity b/w two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


# Create similarity matrix among all sentences
def build_similarity_matrix(sentences, stop_words):
    # create an empty similarity matrix
    similarity_matrix_2 = np.zeros((len(sentences), len(sentences)))
    print(len(sentences))
    for idx1 in range(len(sentences)):
        for idx2 in range(idx1, len(sentences)):
            if idx1 != idx2:
                res = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
                similarity_matrix_2[idx1][idx2] = res
                similarity_matrix_2[idx2][idx1] = res

    return similarity_matrix_2


def generate_summary(text, top_n):
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step1: read text and tokenize
    print('Step1')
    sentences = read_article(text)
    # Step2: generate similarity matrix
    print('Step2')
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    # Step3: Rank sentences in similarity matrix
    print('Step3')
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step4: sort the rank and place top sentences
    print('Step4')
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    # Step5: get the top n number of sentences based on rank
    print('Step5')
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])
    # Step6 : output the summarized version
    print('Step6')
    for i in range(len(summarize_text)):
        print(i+1, "-", summarize_text[i])

    return " ".join(summarize_text), len(sentences)


if __name__ == '__main__':
    for filename in os.listdir('docs/testing/original'):
        # open text file in read mode
        text_file = open('docs/testing/original/' + filename, 'r', encoding='utf-8')
        # read whole file to a string
        data = text_file.read()
        # close file
        text_file.close()
        test = data.split()
        data = " ".join(test)
        sentences = read_article(data)
        if len(sentences) >= 500:
            continue
        summarized, num = generate_summary(data, 5)
        # print(summarized)
        # print(num)
        result_file = 'docs/testing/extracted/' + filename
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(summarized)


