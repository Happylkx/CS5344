import string
import json
# import math
import pyspark as spark
from pyspark import SparkContext, SparkConf
import numpy as np

from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def isfloat(num) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def tokenize_and_filter(line: str, stop_words: set) -> list:
    """
    Tokenize a line of string and remove numbers and latex commands to reduce vocabulary size.

    Parameters:
    - line (str): The input string to tokenize and filter.
    - stop_words (set): A set of stop words to be removed from the tokenized output.

    Returns:
    - list: A list of filtered and tokenized terms.
    """
    # Remove punctuations
    line = line.translate(dict.fromkeys(string.punctuation))
    word_tokens = word_tokenize(line.lower())
    terms = filter(lambda x: x not in stop_words, word_tokens)
    # Filter numbers
    terms = filter(lambda x: not isfloat(x), terms)
    # Filter latex commands \*
    terms = filter(lambda x: not x.startswith("\\"), terms)

    return list(terms)


def tf_idf(tf, n, df):
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) score.

    Parameters:
    tf: Number of times the term(word) appears in the document.
    n: The total number of documents.
    df: The number of documents containing the term(word).

    Returns:
    float: The TF-IDF score for the term(word).
    """
    if tf > 0:
        return (1 + np.log10(tf)) * np.log10(n / df)
    else:
        return 0

def load_stop_words():
    with open("stopwords.txt") as f:
        return set(f.read().split())


def parse_json(line):
    data = json.loads(line)
    return data["id"], data["abstract"]

def init_TF_IDF_corpus(docs, stop_words):
    """ Process the candidates. Calculate term frequencies and document frequencies.
    """
    # Call the .count() method of the RDD to get total count
    # count() is distributed - no need to broadcast.
    num_candidates = docs.count()

    # Parse documents
    # Step 1. Calculate term frequencies
    candidate_doc_words = docs.flatMap(lambda x: [(x[0], word) for word in tokenize_and_filter(x[1], stop_words)])  # Use flatMap to unravel the list. (id, word)
    candidate_doc_words_count = candidate_doc_words.map(lambda x: ((x[0], x[1]), 1))  # ((id, word), 1)
    # print(candidate_doc_words_count.take(1))
    candidate_doc_words_count = candidate_doc_words_count.reduceByKey(lambda x, y: x + y)  # ((id, word), word_count)

    # Step 2. Calculate document frequencies
    inversed_doc_words = candidate_doc_words.map(lambda x: (x[1], x[0])).distinct()  # Inverse and remove duplicates. (word, id)
    inversed_doc_words_count = inversed_doc_words.map(lambda x: (x[0], 1))  # (word, 1)
    inversed_doc_words_count = inversed_doc_words_count.reduceByKey(lambda x, y: x + y)  # (word, doc_count)
    # print(inversed_doc_words_count.take(1))

    # Step 3. Join previous results and compute TF-IDF values in a distributed manner
    word_docid_count = candidate_doc_words_count.map(lambda x: (x[0][1], (x[0][0], x[1])))  # (word, (id, word_count))
    joined_count_data = word_docid_count.join(inversed_doc_words_count)  # (word, ((id, word_count), doc_count))
    # print(joined_count_data.take(3))
    tf_idf_values = joined_count_data.map(lambda x: (x[1][0][0], (x[0], tf_idf(tf=x[1][0][1],
                                                                            n=num_candidates,
                                                                            df=x[1][1]
                                                                        ))))  # (doc_id, (word, tf_idf))
    # print(tf_idf_values.take(3))
    # print(tf_idf_values.collect())

    # Step 4. Process results
    # Compute vector norms
    # Make sure the results reduced are compatible so they can be reduced again!
    normalize_factor = tf_idf_values.map(lambda x: (x[0], x[1][1])).map(lambda x: (x[0], x[1]**2)).reduceByKey(lambda x, y: x+y)  # (doc_id, s^2)
    normalize_factor = normalize_factor.map(lambda x: (x[0], np.sqrt(x[1])))  # (doc_id, s)
    # print(normalize_factor.take(3))
    joint_tf_idf_data = tf_idf_values.join(normalize_factor)  # (doc_id, ((word, tf_idf), s^2))
    # Normalize TF-IDF vectors
    tf_idf_values_normalized = joint_tf_idf_data.map(lambda x: (x[0], (x[1][0][0], x[1][0][1]/x[1][1])))  # (doc_id, (word, tf_idf))
    # print(tf_idf_values_normalized.collect())

    # Collect useful information
    # Build vocabulary id map
    words = sorted(word_docid_count.map(lambda x: x[0]).distinct().collect())
    master_words_map = {word:i for i, word in enumerate(words)}
    # Send to every worker
    words_map_broadcast = sc.broadcast(master_words_map)
    word_map = words_map_broadcast.value
    # print(word_map)
    num_words = len(word_map)

    # Build document frequency counts
    df_counts = np.zeros((len(word_map),))
    for word, doc_count in inversed_doc_words_count.collect():  # [(word, doc_count)]
        df_counts[word_map[word]] = doc_count
    # Send to every worker
    df_counts = sc.broadcast(df_counts)
    df_counts = df_counts.value

    # Build vectors
    def _map_token_list_to_vector(x: tuple, word_map:dict):
        doc_id, lst = x
        # Create an all-zero numpy vector
        ret = np.zeros((len(word_map),))
        # Fill in existing values
        for word, tf_idf_value in lst:
            ret[word_map[word]] = tf_idf_value

        return (doc_id, ret)

    doc_groups = tf_idf_values_normalized.groupByKey()
    doc_words_list = doc_groups.mapValues(list)  # (doc_id, [(word, tf_idf)])
    doc_tf_idf_vectors = doc_words_list.map(lambda x: _map_token_list_to_vector(x=x, word_map=word_map))  # (doc_id, tf_idf_vector)

    return doc_tf_idf_vectors, doc_tf_idf_vectors, word_map, num_candidates, df_counts


def batched_query(queries, docs, doc_tf_idf_vectors, word_map, num_candidates, df_counts, stop_words):
    """
    Given a file of queries and processed candidate docs, this functions finds the docs and sentences with the highest cosine similarity score.
    """
    query_doc_words = queries.map(lambda x: (x[0], tokenize_and_filter(x[1], stop_words)))  # (id, [word])

    def _batched_tf_idf(tf, n, df):
        return np.where(tf != 0, (1 + np.log10(tf)) * np.log10(n / df), 0)

    def _query_normalized_tf_idf(word_list: list):
        assert word_map, "Word map is not initialized!"

        ret = np.zeros((len(word_map),))
        # Get word count
        for word in word_list:
            if word in word_map:
                ret[word_map[word]] += 1

        tf_idf_values = _batched_tf_idf(ret, n=num_candidates, df=df_counts)
        tf_idf_values = tf_idf_values / np.linalg.norm(tf_idf_values)

        return tf_idf_values

    query_tf_idf = query_doc_words.map(lambda x: (x[0], _query_normalized_tf_idf(x[1])))  # (query_id, tf_idf_vector)
    # print(query_tf_idf.collect())
    # doc_tf_idf_vectors: (doc_id, tf_idf_vector)
    # query_tf_idf: (query_id, tf_idf_vector)
    qk_cartesian = query_tf_idf.cartesian(doc_tf_idf_vectors)  # ((query_id, tf_idf_vector), (doc_id, tf_idf_vector))
    cos_similarity = qk_cartesian.map(lambda x: (x[0][0], (x[1][0], np.dot(x[0][1], x[1][1]))))  # (query_id, (doc_id, cos_similarity))


    top_one_cos_similarity = cos_similarity.reduceByKey(lambda x, y: x if x[1]>=y[1] else y)  # (query_id, (doc_id, cos_similarity))
    top_one_cos_similarity_and_tfidf = top_one_cos_similarity.join(query_tf_idf)  # (query_id, ((doc_id, cos_similarity), tf_idf_vector))
    top_one_tfidf = top_one_cos_similarity_and_tfidf.map(lambda x: (x[0], (x[1][0][0], x[1][1])))  # (query_id, (doc_id, tf_idf_vector))

    top_one_cos_similarity_and_text = top_one_tfidf.map(lambda x: (x[1][0], (x[0], x[1][1]))).join(docs)  # (doc_id, ((query_id, tf_idf_vector), abstract_text))

    def _sentence_top_tfidf_match(x):
        """
        Since we know that the abstract is short and can fit on one machine, 
        we can compute sentence tfidf on a single worker.
        """
        doc_id, ((query_id, tf_idf_vector), abstract_text) = x
        sentences = list(filter(len, abstract_text.replace('\n', ' ').split('.')))

        n = len(sentences)
        tfs = np.zeros((n, len(word_map)))
        df = np.zeros((len(word_map),))
        for i, sentence in enumerate(sentences):
            words = tokenize_and_filter(sentence, stop_words=stop_words)
            # Compute term frequency
            for word in words:
                if word in word_map:
                    tfs[i, word_map[word]] += 1

            # Compute document frequency
            for word in set(words):
                if word in word_map:
                    df[word_map[word]] += 1

        
        # Calculate tf-idf values
        sentence_tf_idf_values = _batched_tf_idf(tf=tfs, n=n, df=df)
        normalized_sentence_tf_idf_values = sentence_tf_idf_values / np.expand_dims(np.linalg.norm(sentence_tf_idf_values, axis=1), axis=1)

        # Find top match
        cos_similarity = np.matmul(normalized_sentence_tf_idf_values, np.expand_dims(tf_idf_vector, axis=1)).flatten()  # (n,dim) @ (dim, 1) -> (n, 1)
        # Replace NaNs with 0
        cos_similarity = np.nan_to_num(cos_similarity, nan=0)
        top_idx = np.argmax(cos_similarity)
        top_sentence = sentences[top_idx]
        top_similarity = cos_similarity[top_idx]

        return query_id, (top_sentence, top_similarity)

    top_doc_matches = top_one_cos_similarity  # (query_id, (doc_id, cos_similarity))
    top_sentence_matches = top_one_cos_similarity_and_text.map(_sentence_top_tfidf_match)  # (query_id, (top_sentence, top_sentence_sim_score))

    top_matches = top_doc_matches.join(top_sentence_matches).collect()  # (query_id, ((doc_id, cos_similarity), (top_sentence, top_sentence_sim_score)))

    return top_matches


if __name__ == "__main__":
    conf = SparkConf().setAppName("ArXiv")
    sc = SparkContext(conf=conf)
    # Make sure that the file is sent to every worker
    sc.addFile("stopwords.txt")

    # Load files
    candidate_file = sc.textFile("candidate.jsonl")
    query_file = sc.textFile("query.jsonl")
    stop_words = load_stop_words()

    candidate_abstracts = candidate_file.map(parse_json)
    query_abstracts = query_file.map(parse_json)


    doc_tf_idf_vectors, doc_tf_idf_vectors, word_map, num_candidates, df_counts = init_TF_IDF_corpus(docs=candidate_abstracts, stop_words = stop_words)
    top_matches = batched_query(queries=query_abstracts,
                docs=candidate_abstracts,
                doc_tf_idf_vectors=doc_tf_idf_vectors,
                word_map=word_map,
                num_candidates=num_candidates,
                df_counts=df_counts,
                stop_words=stop_words
                )


    # Output the results
    # <query id> <most relevant abstract id> <abstract relevance score> <most relevance sentence> <sentence relevance score>
    with open("out.txt", 'w') as f:
        for query_result in top_matches:
            query_id, ((doc_id, cos_similarity), (top_sentence, sentence_sim_score)) = query_result
            f.write(f"{query_id} {doc_id} {cos_similarity} {top_sentence} {sentence_sim_score}\n")

    # Stop Spark
    sc.stop()
