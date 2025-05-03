from functools import partial
from .pinecone import vector_store, build_retriever

retriever_map = {
    "pinecone_1": partial(build_retriever, k=1),
    "pinecone_2": partial(build_retriever, k=2),
    "pinecone_5": partial(build_retriever, k=5),
    "pinecone_10": partial(build_retriever, k=10),
}