# %% [markdown]
# ## Preprocessing

# %% [markdown]
# ### Read

# %%
import pandas as pd
import re
import json
import numpy as np

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
df = read_first_n_lines(file_path, 10000)

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
df_short = df.drop_duplicates(subset='title', keep='first')
df_short = df_short.reset_index(drop=True)

# %% [markdown]
# ### Lemmatization

# %%
nlp = spacy.load("en_core_web_sm")
def lemma(docs=df.abstract):
    pip = nlp.pipe(docs, batch_size=32, n_process=-1, disable=["parser", "ner"])
    return [
        [tok.lemma_.lower() for tok in doc if not tok.is_punct and not tok.is_space]
        for doc in tqdm(pip, total=len(docs))
    ]


import os
if not os.path.exists("all_words.pt"):
    all_words = lemma(df.abstract)
    torch.save(all_words, "all_words.pt")
else:
    all_words = torch.load("all_words.pt")
    
if not os.path.exists("all_title_words.pt"):
    all_title_words = lemma(df.title)
    torch.save(all_title_words, "all_title_words.pt")
else:
    all_title_words = torch.load("all_title_words.pt")
    

# %%
# create lists that contain lemmatized words from abstracts, titles, and lists of authors and categories for each entry in the DataFrame

all_authors, all_categories= [], []

for index, row in df_short.iterrows():
    text = row['abstract']
    title = row['title']
    author = row['authors_parsed']
    category = row['categories'] 
    all_authors.append(author)
    all_categories.append(category)

print(all_authors)
print(all_categories)

# %% [markdown]
# ### Delete Stopwords

# %%
nltk.download('stopwords')

# Laden der Stoppwörter
stop_words = set(stopwords.words('english'))

# Entfernen der Stoppwörter aus jedem Dokument
filtered_all_words = []
for words in all_words:
    filtered_words = [word for word in words if word not in stop_words]
    filtered_all_words.append(filtered_words)

# %% [markdown]
# ### List with every word

# %%
# create lists with every value without double lists
if not os.path.exists("hetero_graph_temp.pt"):

    words_values = [word for sublist in filtered_all_words for word in sublist]

    print(words_values)

    authors_values = [author for sublist in all_authors for author in sublist]

    categories_values = [category for sublist in all_categories for category in sublist]

    title_values = [title for sublist in all_title_words for title in sublist]

    # %% [markdown]
    # ### Delete duplicates

    # %%
    categories_list = OrderedSet(categories_values)
    authors_list = OrderedSet(authors_values)
    title_words_list = OrderedSet(title_values)
    words_list = OrderedSet(words_values)

    # %% [markdown]
    # ### Liste mit allen Attributen erstellen

    # %%
    licenses_list = df_short['license'].tolist()

    doi_list = df_short['doi'].tolist()

    title_list = df_short['title'].tolist()

    comment_list = df_short['comments'].tolist()

    journal_list = df_short['journal-ref'].tolist()
    journal_list_set = set(journal_list)

    words_in_title_list = df_short['title'].tolist()

    date_list = df_short['timestamp'].tolist()

    id_list = df_short['id'].tolist()

    pages_list = df_short['pages'].tolist()

    # %% [markdown]
    # ## Create Hetero Object

    # %%
    data = HeteroData()
    data['paper'].num_nodes = len(df_short)
    data['paper'].license = licenses_list
    data['paper'].doi = doi_list
    # data['paper'].title = title_list
    data['paper'].pages = pages_list
    data['paper'].journal = journal_list
    data['paper'].date = date_list
    data['paper'].id = id_list

    data['author'].num_nodes = len(authors_list)
    # data['author'].name = authors_list

    data['category'].num_nodes = len(categories_list)
    # data['category'].name = categories_list

    data['journal'].num_nodes = len(journal_list_set)
    # data['journal'].name = journal_list

    data['word'].num_nodes = len(words_list)
    # data['word'].name = words_list
    torch.save(data, "hetero_graph_temp.pt")

else:
    data = torch.load("hetero_graph_temp.pt")

print('heterograph')

    # %% [markdown]
    # ### Create edges

    # %%
if not os.path.exists("hetero_graph_temp3.pkl"):

    # edge paper written by author
    unique_titles = df_short['title'].unique()
    title_to_id = {title:i for i, title in enumerate(unique_titles)}
    id_to_paper = {i:title for i, title in enumerate(unique_titles)}


    idtitle = df_short['title'].apply(lambda x: title_to_id[x])


    author_to_id = {author:i for i, author in enumerate(authors_list)}
    id_to_author = {i:author for i, author in enumerate(authors_list)}

    data['paper'].title = [id_to_paper[i] for i in range(len(id_to_paper))]
    data['author'].name = [id_to_author[i] for i in range(len(id_to_author))]

    authors_in_paper = [[author_to_id[author] for author in authors] for authors in all_authors]

    edge1 = [[paper_id, author_id] for paper_id, author_list in zip(idtitle, authors_in_paper) for author_id in author_list]
    edge1 = torch.tensor(edge1).T

    data['paper', 'written_by', 'author'].edge_index = edge1

    # edge paper has category
    category_to_id = {category:i for i, category in enumerate(categories_list)}
    id_to_category = {i:category for i, category in enumerate(categories_list)}

    data['category'].name = [id_to_category[i] for i in range(len(id_to_category))]

    categories_in_paper = [[category_to_id[category] for category in categories] for categories in all_categories]

    edge2 = [[paper_id, category_id] for paper_id, category_list in zip(idtitle, categories_in_paper) for category_id in category_list]
    edge2 = torch.tensor(edge2).T

    data['paper', 'has_category', 'category'].edge_index = edge2

    # edge paper has word word
    all_words = words_list.union(title_words_list)
    all_unique_words = OrderedSet(all_words)

    word_to_id = {word:i for i, word in enumerate(all_unique_words)}
    id_to_word = {i:word for i, word in enumerate(all_unique_words)}

    data['word'].name = [id_to_word[i] for i in range(len(id_to_word))]

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

    journal_in_paper = [journal_to_id[journal] if journal in journal_to_id else None for journal in all_unique_journals]

    edge5 = [[paper_id, journal_id] for paper_id, journal_id in zip(idtitle, journal_in_paper)]
    edge5 = torch.tensor(edge5).T

    data['paper', 'in_journal', 'journal-ref'].edge_index = edge5

    print('edges done')
    

    # %% [markdown]
    # ### Assign weights

    # %% [markdown]
    # #### tf-idf: doc-word

    # %%
    # https://medium.com/analytics-vidhya/demonstrating-calculation-of-tf-idf-from-sklearn-4f9526e7e78b

    with open('hetero_graph_temp3.pkl', 'wb') as f:
        pickle.dump(data, f)
else:
    print('skip')
    with open('hetero_graph_temp3.pkl', 'rb') as f:
        data = pickle.load(f)

    # %%
if not os.path.exists("hetero_graph_tfidf.pkl"):
    id_to_paper = {}
    id_to_word = {}

    # creates mapping in order
    word_id = 0

    # Iteriere durch die Zeilen des DataFrames
    for i, words in enumerate(filtered_all_words):
        paper_title = df_short['title'][i]
        if paper_title not in id_to_paper:
            id_to_paper[paper_title] = paper_id
            paper_id += 1

        for word in words:
            if word not in id_to_word:
                id_to_word[word] = word_id
                word_id += 1


    # %%
    # Identitätsfunktion, die als Tokenizer verwendet wird
    def identity_tokenizer(text):
        return text

    # TF-IDF Vektorizer mit der Identitätsfunktion als Tokenizer
    vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, vocabulary=id_to_word)

    # Angenommen, 'filtered_all_words' ist eine Liste von Listen von Token
    tfidf_matrix = vectorizer.fit_transform(filtered_all_words)

    # Initialisierung der Kantenliste und der TF-IDF-Gewichte
    edge_index_list_has_word = [[], []]
    tfidf_weights = []

    # Extraktion der TF-IDF-Werte und Erstellung der Kanten
    for i in range(len(df_short)):
        paper_title = df_short['title'][i]
        paper_idx = id_to_paper[paper_title]

        words = filtered_all_words[i]
        for word in words:
            if word in id_to_word:
                word_idx = id_to_word[word]
                tfidf_weight = tfidf_matrix[i, id_to_word[word]] # FEHLER HIER

                if tfidf_weight:  # Fügen Sie nur Kanten für positive TF-IDF-Werte hinzu
                    edge_index_list_has_word[0].append(paper_idx)
                    edge_index_list_has_word[1].append(word_idx)
                    tfidf_weights.append(tfidf_weight)

    # Konvertiere die Listen in torch.Tensor-Objekte
    edge_index_tensor_has_word = torch.tensor(edge_index_list_has_word, dtype=torch.long)
    tfidf_weights_tensor = torch.tensor(tfidf_weights, dtype=torch.float)

    # Weise die Edge-Indizes und -Attribute dem HeteroData-Objekt zu
    data['paper', 'has_word', 'word'].edge_index = edge_index_tensor_has_word
    data['paper', 'has_word', 'word'].edge_attr = tfidf_weights_tensor

    with open('hetero_graph_tfidf.pkl', 'wb') as f:
        pickle.dump(data, f)
        
else:
    print('skiptfidf')
    with open('hetero_graph_tfidf.pkl', 'rb') as f:
        data = pickle.load(f)

    # %%
    tfidf_df = pd.DataFrame({
        'tfidf_weight': tfidf_weights
    })
    tfidf_df.tfidf_weight.hist(bins=100)

    # %% [markdown]
    # #### PMI: word-word

    # %%
    # https://www.listendata.com/2022/06/pointwise-mutual-information-pmi.html

    # %%
if not os.path.exists("hetero_graph_pmi.pkl"):

    tokens = [item for sublist in filtered_all_words for item in sublist]

    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens, window_size=10)

    # Initialisieren der Kanten und PMI-Werte
    edge_index = [[], []]

    # Berechnen der NPMI-Werte
    npmi_values = []

    for bigram, pmi_score in finder.score_ngrams(bigram_measures.pmi):
        if pmi_score > 0: # instead of npmi > 0
            word1, word2 = bigram

            # calculate normalization
            pxy = finder.ngram_fd[bigram] / finder.N # Berechnen der gemeinsamen Wahrscheinlichkeit p(x, y)
            npmi = pmi_score / -np.log2(pxy) # formula
            # print(f"{finder.ngram_fd[bigram]}, {finder.N}, {pmi_score}, {npmi}")

            # Hinzufügen der berechneten NPMI-Werte und Edge-Index zur Liste
            edge_index[0].append(id_to_word[word1])
            edge_index[1].append(id_to_word[word2])
            npmi_values.append(npmi)

    # Konvertieren in PyTorch Tensoren
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    npmi_values_tensor = torch.tensor(npmi_values, dtype=torch.float)

    # Hinzufügen der Kanten und Kantenattribute zum HeteroData-Objekt
    edge_type = ('word', 'co_occurs_with', 'word')

    data[edge_type].edge_index = edge_index_tensor
    data[edge_type].edge_attr = npmi_values_tensor

    with open('hetero_graph_pmi.pkl', 'wb') as f:
        pickle.dump(data, f)
        
else:
    print('skiptfidf')
    with open('hetero_graph_pmi.pkl', 'rb') as f:
        data = pickle.load(f)

    # %%
    # npmi = pd.DataFrame({"npmi":npmi_values_tensor})
    # npmi.npmi.hist(bins=100)

    # %%
    # finder.score_ngrams(bigram_measures.pmi)

    # %% [markdown]
    # #### Jaccard Similarity for doc-doc

    # %%
    # https://medium.com/@mayurdhvajsinhjadeja/jaccard-similarity-34e2c15fb524
    # https://www.learndatasci.com/glossary/jaccard-similarity/#:~:text=The%20Jaccard%20similarity%20measures%20the,of%20observations%20in%20either%20set.

    # %%
    # Angenommen, 'documents' ist eine Liste von Listen, wobei jede innere Liste die Wortindizes eines Dokuments enthält
    
if not os.path.exists("hetero_graph_jaccard.pkl"):

    documents = filtered_all_words

    # Berechnen Sie die Jaccard-Ähnlichkeit für jedes Dokumentenpaar
    doc_edge_index = [[], []]
    doc_similarity_values = []

    def jaccard_set(list1, list2):
        """Define Jaccard Similarity function for two sets"""
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    for i in tqdm(range(len(documents)), desc="Calculate Jaccard similarity"):
        for j in range(i + 1, len(documents)):
            similarity = jaccard_set(documents[i], documents[j])
            
            # Fügen Sie die Kanten und Ähnlichkeitswerte hinzu
            doc_edge_index[0].append(i)
            doc_edge_index[1].append(j)
            doc_similarity_values.append(similarity)

    # Konvertieren in PyTorch Tensoren
    doc_edge_index_tensor = torch.tensor(doc_edge_index, dtype=torch.long)
    doc_similarity_values_tensor = torch.tensor(doc_similarity_values, dtype=torch.float)

    # Hinzufügen der Kanten und Kantenattribute zum HeteroData-Objekt
    doc_edge_type = ('doc', 'similarity', 'doc')
    data[doc_edge_type].edge_index = doc_edge_index_tensor
    data[doc_edge_type].edge_attr = doc_similarity_values_tensor

    with open('hetero_graph_jaccard.pkl', 'wb') as f:
        pickle.dump(data, f)
        
else:
    print('skiptfidf')
    with open('hetero_graph_jaccard.pkl', 'rb') as f:
        data = pickle.load(f)



    # # %%
    # df_dd = pd.DataFrame({
    #     "jaccard": doc_similarity_values
    # })

    # df_dd.jaccard.hist(bins=100)

    # %%
    # df_dd.jaccard.nlargest(50)

    # %%
    # find duplicate abstracts -> reason for similarity 1.0
    # non_unique_abstracts = df[df.duplicated('abstract', keep=False)]
    # non_unique_abstracts

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


