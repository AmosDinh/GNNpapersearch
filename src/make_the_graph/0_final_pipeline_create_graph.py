# %% [markdown]
# ## Preprocessing

# %% [markdown]
# ### Read

# %%
import pandas as pd
import re
import json
import numpy as np

import os
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

from tqdm.auto import tqdm

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from ordered_set import OrderedSet

# %%
def read_first_n_lines(file_path, n=10000):
    data = []
    with open(file_path, 'r') as file:
        for i, line in tqdm(enumerate(file), total=n):
            if i >= n:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

file_path = 'data/arxiv-metadata-oai-snapshot.json'

# DataFrame erstellen
df = read_first_n_lines(file_path, 1000)
# %%
df.dtypes

# %%
df.head()

# %% [markdown]
# ### Transform columns

# %%
# extract number of pages from string: e.g. "37 pages, 15 figures; published version"

def extract_pages(s):
    match = re.search(r"(\d+)\s*pages", s)
    if match:
        return int(match.group(1))
    else:
        return None

# %%
# format columns to right format and extract information

df['authors_parsed'] = df['authors_parsed'].apply(lambda x: [" ".join(i).strip() for i in (eval(x) if isinstance(x, str) else x)]) # authors to name, first name and list
df['versions'] = df['versions'].apply(lambda x: eval(x)[0]["created"] if isinstance(x, str) else x) # first version / created
df['timestamp'] = df['versions'].apply(lambda x: x[0]['created'] if isinstance(x, list) and len(x) > 0 else None)
df['timestamp'] = pd.to_datetime(df['timestamp'], format="%a, %d %b %Y %H:%M:%S %Z", errors='coerce')
df["categories"] = df["categories"].apply(lambda x: x.split(" ")) # sdeperate categories by comma
df.drop(columns=["submitter", "versions", "update_date", "authors"], inplace=True)
df["pages"] = df.comments.apply(lambda x: extract_pages(str(x))) # extract page number
df.head()

# %% [markdown]
# ### Define size of dataset

# %%
# delete duplicates

df_short = df.drop_duplicates(subset='title', keep='first')
df_short = df_short.reset_index(drop=True)

# %% [markdown]
# ### Lemmatization

# %%
# use lemmatization to get the root of each word of the abstract and title

nlp = spacy.load("en_core_web_sm")
def lemma(docs=df.abstract):
    pip = nlp.pipe(docs, batch_size=32, n_process=1, disable=["parser", "ner"])
    return [
        [tok.lemma_.lower() for tok in doc if not tok.is_punct and not tok.is_space]
        for doc in tqdm(pip, total=len(docs))
    ]

all_words = lemma(df.abstract)
all_title_words = lemma(df.title)

# %%
# create lists that contain all entries of authors and categories

all_authors, all_categories= [], []

for index, row in df_short.iterrows():
    author = row['authors_parsed']
    category = row['categories'] 
    all_authors.append(author)
    all_categories.append(category)

# %% [markdown]
# ### Delete Stopwords

# %%
# delete stopwords in abstract and title

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

filtered_all_words = []
for words in all_words:
    filtered_words = [word for word in words if word not in stop_words]
    filtered_all_words.append(filtered_words)
    
filtered_title_words = []
for words in all_title_words:
    filtered_words = [word for word in words if word not in stop_words]
    filtered_title_words.append(filtered_words)
    
# %% [markdown]
# ### List with every word

# %%
# create lists with every value 
words_values = [word for sublist in filtered_all_words for word in sublist]

authors_values = [author for sublist in all_authors for author in sublist]

categories_values = [category for sublist in all_categories for category in sublist]

title_values = [title for sublist in all_title_words for title in sublist]

# %% [markdown]
# ### Delete duplicates

# %%
# use orderedset to get unique values while preserving order
# https://www.geeksforgeeks.org/python-ordered-set/

categories_list = OrderedSet(categories_values)
authors_list = OrderedSet(authors_values)
title_words_list = OrderedSet(title_values)
words_list = OrderedSet(words_values)
if not os.path.exists("hetero_graph_temp.pt"):
    # %% [markdown]
    # ### create list with all attributes

    # %%
    licenses_list = df_short['license'].tolist()

    doi_list = df_short['doi'].tolist()

    title_list = df_short['title'].tolist()

    comment_list = df_short['comments'].tolist()

    journal_list = df_short['journal-ref'].tolist()

    words_in_title_list = df_short['title'].tolist()

    date_list = df_short['timestamp'].tolist()

    id_list = df_short['id'].tolist()

    pages_list = df_short['pages'].tolist()

    # %% [markdown]
    # ## Create Hetero Object

    # %%
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html
    # create the graph using heterodata from pytorch geometric
    
    data = HeteroData()
    data['paper'].license = licenses_list
    data['paper'].doi = doi_list
    data['paper'].pages = pages_list
    data['paper'].journal = journal_list
    data['paper'].date = date_list
    data['paper'].id = id_list

    torch.save(data, "hetero_graph_temp.pt")
else:
    data = torch.load("hetero_graph_temp.pt")

print('heterograph')

    # %% [markdown]
    # ### Create edges

    # %%

# edge paper written by author
unique_titles = df_short['title'].unique()


id_to_paper = {i:title for i, title in enumerate(unique_titles)}

if not os.path.exists("hetero_graph_temp3.pkl"):
    # create dicts to map title to id and id to title
    title_to_id = {title:i for i, title in enumerate(unique_titles)}
    idtitle = df_short['title'].apply(lambda x: title_to_id[x])

    # create dicts to map author to id and id to author
    author_to_id = {author:i for i, author in enumerate(authors_list)}
    id_to_author = {i:author for i, author in enumerate(authors_list)}
    
    # create list of names and authors for each paper and add data to graph
    data['paper'].name = [id_to_paper[i] for i in range(len(id_to_paper))]
    data['paper'].num_nodes = len(data['paper'].name)
    data['author'].name = [id_to_author[i] for i in range(len(id_to_author))]
    data['author'].num_nodes = len(data['author'].name)
    print('paper author')

    # list of lists with authors for each paper
    authors_in_paper = [[author_to_id[author] for author in authors] for authors in all_authors]

    # create edge by using two lists of same length, mapping the paper id to the author id
    edge1 = [[paper_id, author_id] for paper_id, author_list in zip(idtitle, authors_in_paper) for author_id in author_list]
    edge1 = torch.tensor(edge1).T

    # add edge to graph
    data['paper', 'written_by', 'author'].edge_index = edge1

    # edge paper has category
    category_to_id = {category:i for i, category in enumerate(categories_list)}
    id_to_category = {i:category for i, category in enumerate(categories_list)}

    data['category'].name = [id_to_category[i] for i in range(len(id_to_category))]
    data['category'].num_nodes = len(data['category'].name)

    categories_in_paper = [[category_to_id[category] for category in categories] for categories in all_categories]

    edge2 = [[paper_id, category_id] for paper_id, category_list in zip(idtitle, categories_in_paper) for category_id in category_list]
    edge2 = torch.tensor(edge2).T

    data['paper', 'has_category', 'category'].edge_index = edge2

    # edge paper has word word
all_words = words_list.union(title_words_list)
all_unique_words = OrderedSet(all_words)

word_to_id = {word:i for i, word in enumerate(all_unique_words)}

if not os.path.exists("hetero_graph_temp3.pkl"):
    id_to_word = {i:word for i, word in enumerate(all_unique_words)}
   
    data['word'].name = [id_to_word[i] for i in range(len(id_to_word))]
    data['word'].num_nodes = len(data['word'].name)
    print('words in paper')
    words_in_paper = [[word_to_id[word] for word in words] for words in filtered_all_words]	
 
    edge3 = [[paper_id, word_id] for paper_id, word_list in zip(idtitle, words_in_paper) for word_id in word_list]
    edge3 = torch.tensor(edge3).T

    data['paper', 'has_word', 'word'].edge_index = edge3

    # edge paper has titleword word
    words_in_title = [[word_to_id[word] for word in title] for title in filtered_title_words] 
    
    edge4 = [[paper_id, word_id] for paper_id, word_list in zip(idtitle, words_in_title) for word_id in word_list]
    edge4 = torch.tensor(edge4).T 

    data['paper', 'has_titleword', 'word'].edge_index = edge4

    # edge paper has journal journal
    all_unique_journals = OrderedSet(journal_list)

    journal_to_id = {journal:i for i, journal in enumerate(all_unique_journals)}
    id_to_journal = {i:journal for i, journal in enumerate(all_unique_journals)}
    
    data['journal'].name = [id_to_journal[i] for i in range(len(id_to_journal))]
    data['journal'].num_nodes = len(data['journal'].name)
    print('journal in paper')
    journal_in_paper = [journal_to_id[journal] if journal in journal_to_id else None for journal in journal_list]

    edge5 = [[paper_id, journal_id] for paper_id, journal_id in zip(idtitle, journal_in_paper)]
    edge5 = torch.tensor(edge5).T

    data['paper', 'in_journal', 'journal'].edge_index = edge5
    print('edges done')
    
    # %% [markdown]
    # ### Assign weights

    # %% [markdown]
    # #### tf-idf: doc-word

    # %%
    # https://medium.com/analytics-vidhya/demonstrating-calculation-of-tf-idf-from-sklearn-4f9526e7e78b
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    with open('hetero_graph_temp3.pkl', 'wb') as f:
        pickle.dump(data, f)
else:
    print('skip')
    with open('hetero_graph_temp3.pkl', 'rb') as f:
        data = pickle.load(f)

    # %%
if not os.path.exists("hetero_graph_tfidf.pkl"):
    
    # %%
    # don't tokenize again
    def identity_tokenizer(text):
        return text

    # tf-idf vectorizer sklearn
    vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, vocabulary=word_to_id)
    print('a')
    tfidf_matrix = vectorizer.fit_transform(filtered_all_words)
    print('b')

    # create lists to store values
    edge_index_list_has_word = [[], []]
    print('c')
    tfidf_weights = []
    print('e')

    len_index = data['paper', 'has_word', 'word'].edge_index.shape[1]
    
    # get weights
    tfidf_weights = [tfidf_matrix[data['paper', 'has_word', 'word'].edge_index[0][i], data['paper', 'has_word', 'word'].edge_index[1][i]] for i in tqdm(range(len_index))]
    print('d')

    # create tensor and assign weights to edges
    tfidf_weights_tensor = torch.tensor(tfidf_weights, dtype=torch.float)
    print('e')
    data['paper', 'has_word', 'word'].edge_attr = tfidf_weights_tensor
    print('f')
    with open('hetero_graph_tfidf.pkl', 'wb') as f:
        pickle.dump(data, f)
        
else:
    print('skiptfidf')
    with open('hetero_graph_tfidf.pkl', 'rb') as f:
        data = pickle.load(f)


    # %%
    # https://www.listendata.com/2022/06/pointwise-mutual-information-pmi.html

    # %%
if not os.path.exists("hetero_graph_pmi.pkl"):

    # find bigrams with window_size=10
    tokens = [item for sublist in filtered_all_words for item in sublist]
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens, window_size=10)

    # create lists to store values and connected words
    edge_index = [[], []]
    npmi_values = []

    for bigram, pmi_score in tqdm(finder.score_ngrams(bigram_measures.pmi)):
        if pmi_score > 0:
            word1, word2 = bigram

            # calculate normalization
            pxy = finder.ngram_fd[bigram] / finder.N
            npmi = pmi_score / -np.log2(pxy)

            a = word_to_id[word1]
            b = word_to_id[word2]
            if a == b:
                continue
            edge_index[0].append(a)
            edge_index[1].append(b)
            npmi_values.append(npmi)

    # convert to pytorch tensor
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    npmi_values_tensor = torch.tensor(npmi_values, dtype=torch.float)

    # create edges and assign weights
    edge_type = ('word', 'co_occurs_with', 'word')
    data[edge_type].edge_index = edge_index_tensor
    data[edge_type].edge_attr = npmi_values_tensor

    with open('hetero_graph_pmi.pkl', 'wb') as f:
        pickle.dump(data, f)
        
else:
    print('skiptfidf')
    with open('hetero_graph_pmi.pkl', 'rb') as f:
        data = pickle.load(f)

# %% [markdown]
# # Export

# %%
print("save final")
# Serialize and save the hetero-object
with open('hetero_graph_final.pkl', 'wb') as f:
    pickle.dump(data, f)

# %%
# Load the hetero-object
with open('hetero_graph_final.pkl', 'rb') as f:
    loaded_hetero_graph = pickle.load(f)
loaded_hetero_graph["author"]


