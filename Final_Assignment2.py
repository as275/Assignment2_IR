import os
import math
from collections import defaultdict, Counter

class VectorSpaceModel:
    def __init__(self):
        self.inverted_index = defaultdict(list)  
        self.doc_lengths = defaultdict(float)    
        self.num_documents = 0                   

    def index_document(self, doc_id, document):
        """
        Indexes a document by updating the inverted index and computing document length.
        """
        self.num_documents += 1
        term_freqs = Counter(document)  
        doc_length = 0

        for term, tf in term_freqs.items():
            log_tf = 1 + math.log10(tf)  
            self.inverted_index[term].append((doc_id, log_tf))
            doc_length += log_tf ** 2  

        self.doc_lengths[doc_id] = math.sqrt(doc_length)  

    def compute_idf(self, term):
        """
        Compute inverse document frequency (idf) for a given term.
        """
        df = len(self.inverted_index[term]) 
        if df == 0:
            return 0
        return math.log10(self.num_documents / df)

    def rank_documents(self, query):
        """
        Rank documents based on cosine similarity between the query and document vectors.
        """
        query_terms = Counter(query)
        query_vector = {}
        query_length = 0

        
        for term, tf in query_terms.items():
            if term in self.inverted_index:
                log_tf = 1 + math.log10(tf)
                idf = self.compute_idf(term)
                query_tf_idf = log_tf * idf
                query_vector[term] = query_tf_idf
                query_length += query_tf_idf ** 2

        query_length = math.sqrt(query_length)

        
        scores = defaultdict(float)
        for term, query_weight in query_vector.items():
            if term in self.inverted_index:
                for doc_id, doc_tf in self.inverted_index[term]:
                    scores[doc_id] += query_weight * doc_tf

        
        for doc_id in scores:
            if self.doc_lengths[doc_id] > 0 and query_length > 0:
                scores[doc_id] /= (self.doc_lengths[doc_id] * query_length)

        
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        return ranked_docs[:10] 

def load_documents_from_folder(folder_path):
    
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents[filename] = content.split()  
    return documents


vsm = VectorSpaceModel()


folder_path = r'C:\Users\Akshat\Desktop\IR_Assignment_1\Assignment 2\Corpus'


documents = load_documents_from_folder(folder_path)


for doc_id, doc_text in documents.items():
    vsm.index_document(doc_id, doc_text)


query1 = 'developing your Zomato business account and profile is a great way to boost your restaurant’s online reputation'.lower().split()
print("Q1 Output:", vsm.rank_documents(query1))


query2 = 'Warwickshire came from an ancient family and was the heiress to some land'.lower().split()
print("Q2 Output:", vsm.rank_documents(query2))
