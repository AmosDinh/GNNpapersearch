# GNNpapersearch

### Summary
1. We train a Graph Neural Network on the ArXiv paper dataset which comprises roughly 2 million papers and 1 million authors. 
2. The GNN is used to obtain numerical representations of papers, authors, categories, journals and words. Search is powerered using the Weaviate vector database. 
3. The training approach allows us to compare every entity type with every other one. For example we can find similar papers given a query of an author, a journal or a category.

### Training
The graph is constructed using all entity- and relationship types. As [1] propose,  we create a node for every word and connect them by TF-IDF and PMI-weighted edges to the papers' abstracts they appear in.   

Training is done with the Heterogeneous Graph Transformer [2], corresponding sampling approach and a TransE Knowledge Graph Embedding [3] head with margin loss:
![image](https://github.com/AmosDinh/GNNpapersearch/assets/39965380/4efa8e29-4b70-4784-b014-f3c44fc25f9a)
where s, e, t are the sourcenodes, relationships and targetnodes of all relationship types in the graph e.g (paper written_by author).  <br>
The training takes 36 hours on a P100 Nvidia GPU. At the end of training the GNN still has only seen 1 million target edges (but more than a billion nodes).

### Results
- We create our own qualitative benchmark by specifying a query paper and the result we want to obtain and get the rank of the item we want to be ranked highly (only technical/ml domain).
- Comparing against PCA-reduced TF-IDF, the approach achieves a mean rank of 90,000 out of 2.3 million compared to 190,000 out of 2.3 million (TF-IDF approach). 
- With the project we further show the feasability of performing fast nearest neighbor search for 7 million embeddings on consumer grade hardware (32GB RAM) with the help of Weaviates built-in PQ-quantization [4].


### Sources
[1] Yao, L., Mao, C., & Luo, Y. (2019, July). Graph convolutional networks for text classification. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 7370-7377). <br>
[2] Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020, April). Heterogeneous graph transformer. In Proceedings of the web conference 2020 (pp. 2704-2710). <br>
[3] Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. Advances in neural information processing systems, 26. <br>
[4] https://weaviate.io/blog/ann-algorithms-hnsw-pq#hnswpq




## Running data preprocessing, training and embedding generation.
- src/make_the_graph/data/ contains only 1000 rows of the 3 million row .json. The original link to download the whole dataset can be found in src/make_the_graph/dataset.

Requirements for the python kernel can be installed manually by copying the commands out of requirements.ipynb or running the requirements.ipynb with the correct kernel attached.
The python file and notebooks can be run sequentially in order 0_, 1__, 1_ .. 6_ (The python file should be run from folder src/make_the_graph).
The output will be the gnn embeddings named embeddings_{node_type}.pt,
relationship embeddings relationship_embedding_{rel_type}.pt
and the tfidf embeddings named tfidf_embeddings.pkl.
These pickle files together with the arxiv_author_paper_graph_no_features_bare.pkl which is also produced through the notebooks are then loaded into the weaviate database.

Note: 
- The GNN will only run for one minibatch for demonstration purposes, the break statement must be commented out for longer training.
- 1_serializer.ipynb was used to split up the dataset to train on a machine where the whole pickle could not be loaded at once. It is not necessary for the demonstration.

## Database and Frontend
Start the database and Frontendserver with the command:
```
docker-compose up -d 
```

To fill the database go to
```
cd src/db
python fill_db.py
```

The Frontend can be acceced via browser with this link [http://localhost](http://localhost)

## Benchmark
Benchmark files can be found in src/benchmark
