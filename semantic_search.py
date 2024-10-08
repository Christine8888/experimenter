import json
from typing import List, Dict, Tuple
from collections import Counter
from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm
import yaml
from openai import OpenAI
from filters import CitationFilter, DateFilter, KeywordFilter
from temporal import analyze_temporal_query
import anthropic
from vector_store import Document
from evaluate import RetrievalSystem, main as evaluate_main
from vector_store import EmbeddingClient, Document, DocumentLoader

class EmbeddingRetrievalSystem(RetrievalSystem):
    def __init__(self, embeddings_path: str = "vector_store/embeddings_matrix.npy", 
                 documents_path: str = "vector_store/documents.pkl", 
                 index_mapping_path: str = "vector_store/index_mapping.pkl",
                 metadata_path: str = "vector_store/metadata.json", weight_citation = False, weight_date = False, weight_keywords = False):
        
        self.embeddings_path = embeddings_path
        self.documents_path = documents_path
        self.index_mapping_path = index_mapping_path
        self.metadata_path = metadata_path
        self.weight_citation = weight_citation
        self.weight_date = weight_date
        self.weight_keywords = weight_keywords

        self.embeddings = None
        self.documents = None
        self.index_mapping = None
        self.metadata = None
        self.document_dates = []
        
        self.load_data()
        # self.init_filters()

        try:
            config = yaml.safe_load(open('../config.yaml', 'r'))
            self.set_clients(config['openai_api_key'], config['anthropic_api_key'])
        except:
            self.client = None
            self.anthropic_client = None
            print('API clients not yet set')
    
    def set_clients(self, openai_api_key, anthropic_api_key):
        print('Setting API clients')
        self.client = EmbeddingClient(OpenAI(api_key=openai_api_key))
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    def generate_metadata(self):
        astro_meta = load_dataset("JSALT2024-Astro-LLMs/astro_paper_corpus", split = "train")
        keys = list(astro_meta[0].keys())
        keys.remove('abstract')
        keys.remove('introduction')
        keys.remove('conclusions')

        self.metadata = {}
        for paper in astro_meta:
            id_str = paper['arxiv_id']
            self.metadata[id_str] = {key: paper[key] for key in keys}
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
            print("Wrote metadata to {}".format(self.metadata_path))

    def load_data(self):
        print("Loading embeddings...")
        self.embeddings = np.load(self.embeddings_path)
        
        print("Loading documents...")
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print("Loading index mapping...")
        with open(self.index_mapping_path, 'rb') as f:
            self.index_mapping = pickle.load(f)
        
        print("Processing document dates...")
        self.document_dates = {doc.id: self.parse_date(doc.arxiv_id) for doc in self.documents}
        
        # if os.path.exists(self.metadata_path):
        #     print("Loading metadata...")
        #     with open(self.metadata_path, 'r') as f:
        #         self.metadata = json.load(f)
        #     print("Loaded metadata.")
        # else:
        #     print("Could not find path; generating metadata.")
        #     self.generate_metadata()
        
        print("Data loaded successfully.")
    
    def init_filters(self):
        print("Loading filters...")
        self.citation_filter = CitationFilter(metadata = self.metadata)
        
        self.date_filter = DateFilter(document_dates = self.document_dates)
        
        self.keyword_filter = KeywordFilter(index_path = "../data/vector_store/keyword_index.json", metadata = self.metadata, remove_capitals = True, ne_only = True)

    def retrieve(self, query: str, arxiv_id: str = None, top_k: int = 10, return_scores = False, time_result = None) -> List[Tuple[str, str, float]]:
        query_date = self.parse_date(arxiv_id)
        query_embedding = self.get_query_embedding(query)
        
        # Judge time relevance
        if time_result is None:
            if self.weight_date: time_result, time_taken = analyze_temporal_query(query, self.anthropic_client)
            else: time_result = {'has_temporal_aspect': False, 'expected_year_filter': None, 'expected_recency_weight': None}

        top_results = self.rank_and_filter(query, query_embedding, query_date, top_k, return_scores = return_scores, time_result = time_result)
        
        return top_results
    
    def rank_and_filter(self, query, query_embedding: np.ndarray, query_date, top_k: int = 10, return_scores = False, time_result = None) -> List[Tuple[str, str, float]]:
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Filter and rank results
        if self.weight_keywords: 
            keyword_matches = self.keyword_filter.filter(query)

        
        results = []
        for doc_id, mappings in self.index_mapping.items():
            #if not self.weight_keywords or doc_id in keyword_matches:
            # print("Doc ID: ", doc_id)
            # print("Mappings: ", mappings)
            abstract_sim = similarities[mappings['abstract']] if 'abstract' in mappings else -np.inf
            conclusions_sim = similarities[mappings['conclusions']] if 'conclusions' in mappings else -np.inf
            
            if abstract_sim > conclusions_sim: 
                results.append([doc_id, "abstract", abstract_sim])
            else: 
                results.append([doc_id, "conclusions", conclusions_sim])
                
        
        # Sort and weight and get top-k results
        # if time_result['has_temporal_aspect']:
        #     filtered_results = self.date_filter.filter(results, boolean_date = time_result['expected_year_filter'], time_score = time_result['expected_recency_weight'], max_date = query_date)
        # else:
        #     filtered_results = self.date_filter.filter(results, max_date = query_date)
        
        # if self.weight_citation: self.citation_filter.filter(filtered_results)

        filtered_results = results

        top_results = sorted(filtered_results, key=lambda x: x[2], reverse=True)[:top_k]

        if return_scores:
            return {doc[0]: doc[2] for doc in top_results}

        # Only keep the document IDs
        top_results = [doc[0] for doc in top_results]
        return top_results

    def get_query_embedding(self, query: str) -> np.ndarray:
        embedding = self.client.embed(query)
        return np.array(embedding, dtype = np.float32)
    
    def get_document_texts(self, doc_ids: List[str]) -> List[Dict[str, str]]:
        results = []
        for doc_id in doc_ids:
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                results.append({
                    'id': doc.id,
                    'abstract': doc.abstract,
                    'conclusions': doc.conclusions
                })
            else:
                print(f"Warning: Document with ID {doc_id} not found.")
        return results
    
    def retrieve_context(self, query, top_k, sections = ["abstract", "conclusions"], **kwargs):
        docs = self.retrieve(query, top_k = top_k, return_scores = True, **kwargs)
        docids = docs.keys()
        doctexts = self.get_document_texts(docids) # avoid having to do this repetitively?
        context_str = ""
        doclist = []
        
        for docid, doctext in zip(docids, doctexts):
            for section in sections:
                context_str += f"{docid}: {doctext[section]}\n"
            
            meta_row = self.metadata[docid]
            doclist.append(Document(docid, doctext['abstract'], doctext['conclusions'], docid, title = meta_row['title'],
                                    score = docs[docid], n_citation = meta_row['citation_count'], keywords = meta_row['keyword_search']))

        return context_str, doclist

def main():
    retrieval_system = EmbeddingRetrievalSystem("charlieoneill/jsalt-astroph-dataset")
    # evaluate_main(retrieval_system, "BaseSemanticSearch")
    query = "What is the stellar mass of the Milky Way?"
    arxiv_id = "2301.00001"
    top_k = 10
    results = retrieval_system.retrieve(query, arxiv_id, top_k)
    print(f"Retrieved documents: {results}")

if __name__ == "__main__":
    main()
