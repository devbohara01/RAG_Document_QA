def retrieve_documents(vector_store, query):
    return vector_store.similarity_search_with_score(
        query,
        k=3
    )