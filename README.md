# GNNpapersearch

## Running data preprocessing, training and embedding generation.
Requirements for the python kernel can be installed manually py copying the commands out of requirements.ipynb or running the requirements.ipynb with the correct kernel attached.
Python file and notebooks can be run sequentially in order 0_, 1__, 1_ .. 6_.
The output will be the gnn embeddings named embeddings_{node_type}.pt,
relationship embeddings relationship_embedding_{rel_type}.pt
and the tfidf embeddings named tfidf_embeddings.pkl.
These pickle files together with the arxiv_author_paper_graph_no_features_bare.pkl which is also produced through the notebooks are then loaded into the weaviate database.

