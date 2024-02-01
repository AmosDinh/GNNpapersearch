# GNNpapersearch
Four university student project.
We train a Graph Neural Network on the ArXiv paper dataset which comprises roughly 7 million entities, including 2 million papers and 1 million authors.
We use the GNN to obtain numerical representations of papers, authors, categories, journals and words and power search using the Weaviate Database.

Training is done with the Heterogeneous Graph Transformer, corresponding sampling approach and a TransE Knowledge Graph Embedding Head with margin loss.
The training takes 36 hours on a P100 Nvidia GPU. 

We create our own qualitative benchmark by specifying a query paper and the result we want to obtain and then compute the mean rank of the item we want to be ranked highly.
Comparing against PCA-reduced TF-IDF, 



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
