# %% [markdown]
# ## Preprocessing
# gcloud compute scp amosd@instance-3:/home/amosd/heero_graph_final.pkl.gz heero_graph_final.pkl
# %% [markdown]
# ### Read
import pickle
# %%
import pandas as pd
import re
import spacy
import os
import torch
from torch_geometric.data import HeteroData

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# %%
import json
from tqdm.auto import tqdm 

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


# sudo /home/amosd/miniconda3/envs/gnnpapersearch/bin/python pipeline_create_graph.py 
# DataFrame erstellen
df = read_first_n_lines(file_path, 1000000000000000)


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
# Quelle warum genau die neuen Spalten fürPaper suchen
# df['authors_parsed'] = df['authors_parsed'].apply(lambda x: [" ".join(author).strip() for author in (eval(x) if isinstance(x, str) else x)])

df['authors_parsed'] = df['authors_parsed'].apply(lambda x: [" ".join(i).strip() for i in (eval(x) if isinstance(x, str) else x)]) # authors to name, first name and list
df['versions'] = df['versions'].apply(lambda x: eval(x)[0]["created"] if isinstance(x, str) else x) # first version / created
df['data'] = df['versions'].apply(lambda x: x[0]['created'] if isinstance(x, list) and len(x) > 0 else None)
df['timestamp'] = pd.to_datetime(df['timestamp'], format="%a, %d %b %Y %H:%M:%S %Z", errors='coerce')
df["categories"] = df["categories"].apply(lambda x: x.split(" ")) # sdeperate categories by comma
df.drop(columns=["submitter", "versions", "update_date", "authors"], inplace=True)
df["pages"] = df.comments.apply(lambda x: extract_pages(str(x))) # extract page number
df.head()


# %% [markdown]
# ### Define size of dataset

# %%

# %% [markdown]
# ### Lemmatization

# %%
nlp = spacy.load("en_core_web_sm")

def lemm(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

# %%
import spacy
from tqdm.auto import tqdm

nlp = spacy.load("en_core_web_sm")
def lemma(docs=df.abstract):
    pip = nlp.pipe(docs, batch_size=32, n_process=-1, disable=["parser", "ner"])
    return [
        [tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space]
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


# all_words, all_authors, all_categories, all_title_words = [], [], [], []
all_authors, all_categories= [], []


for index, row in df.iterrows():
    text = row['abstract']
    title = row['title']
    author = row['authors_parsed']
    category = row['categories'] 
    # tokenized_words = tokenize_and_normalize(text)
    # lemm_words = lemm(text)
    # lemm_title_words = lemm(title)
    # tokenized_words_list.append(tokenized_words)
    # all_words.append(lemm_words)
    all_authors.append(author)
    all_categories.append(category)
    # all_title_words.append(lemm_title_words)


##print(all_authors)
#print(all_categories)

# %% [markdown]
# ### Delete Stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Laden der Stoppwörter
stop_words = set(stopwords.words('english'))

# Entfernen der Stoppwörter aus jedem Dokument
filtered_all_words = []
for words in all_words:
    filtered_words = [word for word in words if word not in stop_words]
    filtered_all_words.append(filtered_words)
# %%
# !pip3 install nltk
if not os.path.exists("hetero_graph_temp.pt"):
    # %%
    

    # %%
    stop_words

    # %%
    filtered_all_words

    # %% [markdown]
    # ### List with every word

    # %%
    # create lists with every value without double lists

    words_values = [word.lower() for sublist in filtered_all_words for word in sublist]

    #print(words_values)

    authors_values = [author.lower() for sublist in all_authors for author in sublist]

    categories_values = [category.lower() for sublist in all_categories for category in sublist]

    title_values = [title.lower() for sublist in all_title_words for title in sublist]

    # %% [markdown]
    # ### Delete duplicates

    # %%
    # list with all words without duplicates

    print('making word lists')
    # words_list = []
    # for i in tqdm(range(len(words_values))):
    #     if words_values[i] not in words_list:
    #         words_list.append(words_values[i].lower())


    # # list(set(words_values))
    # print('x1')
    # authors_list = []
    # for i in tqdm(range(len(authors_values))):
    #     if words_values[i] not in authors_list:
    #         authors_list.append(authors_values[i])

    # # list(set(words_values))
    # print('x2')
    # categories_list = []
    # for i in tqdm(range(len(categories_values))):
    #     if words_values[i] not in categories_list:
    #         categories_list.append(categories_values[i].lower())

    # # list(set(words_values))
    # print('x3')
    # title_words_list = []
    # for i in tqdm(range(len(title_values))):
    #     if title_values[i] not in title_words_list:
    #         title_words_list.append(title_values[i].lower())


    # %%
    a1 = set(authors_values)

    # %%
    # print(words_list)
    # print(title_words_list)
    # print(authors_list)
    # print(categories_list)
    # print(len(words_list))

    # %%
    categories_list = set(categories_values)
    authors_list = set(authors_values)
    title_words_list = set(title_values)
    words_list = set(words_values)

    # %%
    print(len(words_list))

    # %% [markdown]
    # ### Liste mit allen Attributen erstellen

    # %%
    # funtioniert, erstellt aber keine listen

    # licenses_list = []
    # def licenses_in_list(df):
    #     licenses_list = df['license'].tolist()
    #     print(licenses_list)

    # doi_list = []
    # def doi_in_list(df):
    #     doi_list = df['doi'].tolist()
    #     print(doi_list)

    # title_list = []
    # def titles_in_list(df):
    #     title_list = df['title'].tolist()
    #     print(title_list)

    # # comment_list = []
    # # def comments_in_list(df):
    # #     for i in tqdm(range(len(df)):
    # #         comment_list.append(df.comments[i])
    # #     print(comment_list)

    # pages_list = []
    # def pages_in_list(df):
    #     pages_list = df['pages'].tolist()
    #     print(pages_list)

    # journal_list = []
    # def journals_in_list(df):
    #     journal_list = df['journal-ref'].tolist()
    #     print(journal_list)

    # date_list = []
    # def date_in_list(df):
    #     date_list = df['timestamp'].tolist()
    #     print(date_list)

    # id_list = []
    # def id_to_list(df):
    #     id_list = df['id'].tolist()
    #     print(id_list)


    # %%
    # create lists of attributes 
    print('create lists of attributes ')
    licenses_list = df['license'].tolist()
    # def licenses_in_list(df):
    #     for i in tqdm(range(len(df))):
    #         licenses_list.append(df.license[i])
    #     print(licenses_list)

    doi_list = df['doi'].tolist()
    # def doi_in_list(df):
    #     for i in tqdm(range(len(df))):
    #         doi_list.append(df.doi[i])
    #     print(doi_list)

    title_list = df['title'].tolist()
    # def titles_in_list(df):
    #     for i in tqdm(range(len(df))):
    #         title_list.append(df.title[i])
    #     print(title_list)

    comment_list = df['comments'].tolist()
    # def comments_in_list(df):
    #     for i in tqdm(range(len(df))):
    #         comment_list.append(df.comments[i])
    #     print(comment_list)


    # author_list = []
    # def authors_in_list(df):
    #     for i in tqdm(range(len(df)):
    #         author_list.append(df.authors_parsed[i])
    #     print(author_list)


    # categories_list = []
    # def categories_in_list(df):
    #     for i in tqdm(range(len(df)):
    #         categories_list.append(df['categories'][i])
    #     print(categories_list)


    journal_list = df['journal-ref'].tolist()

    # def journals_in_list(df):
    #     for i in tqdm(range(len(df))):
    #         journal_list.append(df['journal-ref'][i])
    #     print(journal_list)


    words_in_title_list = df['title'].tolist()
    # def words_in_title(df):
    #     for i in tqdm(range(len(df))):
    #         words_in_title_list.append(df.title[i].split())
    #     print(words_in_title_list)

    date_list = df['timestamp'].tolist()
    # def date_in_list(df):
    #     for i in tqdm(range(len(df))):
    #         date_list.append(df.timestamp[i])
    #     print(date_list)

    id_list = df['id'].tolist()
    # def id_to_list(df):
    #     for i in tqdm(range(len(df))):
    #         id_list.append(df.id[i])
    #     print(id_list)

    pages_list = df['pages'].tolist()
    # def pages_in_list(df):
    #     for i in tqdm(range(len(df))):
    #         pages_list.append(df.pages[i])
    #     print(pages_list)



    # %%
    # print('x8')

    # licenses_in_list(df)
    # print('x89')

    # doi_in_list(df)
    # print('x10')

    # titles_in_list(df)
    # print('x11')

    # pages_in_list(df)
    # # authors_in_list(df)
    # # categories_in_list(df)
    # print('x12')

    # journals_in_list(df)
    # print('x13')

    # date_in_list(df)
    # print('x14')

    # id_to_list(df)

    # %%
    # licenses_list

    # %% [markdown]
    # ## Words to numbers

    # %%
    # # Erstellen eines Mappings von Wort zu Index
    # word_to_index = {word: idx for idx, word in enumerate(set(list_of_words))}

    # # Konvertieren der Wörter in Indizes
    # words_as_numbers = [word_to_index[word] for word in list_of_words]

    # # Konvertieren in einen PyTorch Tensor
    # words_tensor = torch.tensor(words_as_numbers, dtype=torch.long)

    # # Weise die Edge-Indizes dem HeteroData-Objekt zu
    # data['paper', 'written_by', 'author'].edge_index = words_tensor

    # %%
    # data['paper', 'written_by', 'author'].edge_index

    # %%


    # %%
    # # words to numbers

    # author_to_index_written_by = {author: idx for idx, author in enumerate(set(list_of_authors))}
    # edge_index_list_written_by = [
    #     [author_to_index_written_by[author] for author in list_of_authors]
    # ]

    # # Konvertiere die Liste in torch.Tensor-Objekte
    # edge_index_tensor_written_by = torch.tensor(edge_index_list_written_by, dtype=torch.long)

    # # Weise die Edge-Indizes dem HeteroData-Objekt zu
    # data['paper', 'written_by', 'author'].edge_index = edge_index_tensor_written_by

    # %%
    # # Kategorien in Nummern umwandeln
    # category_to_index_has_category = {category: idx for idx, category in enumerate(set(list_of_categories))}
    # edge_index_list_has_category = [
    #     [category_to_index_has_category[category] for category in list_of_categories]
    # ]

    # # Konvertiere die Liste in torch.Tensor-Objekte
    # edge_index_tensor_has_category = torch.tensor(edge_index_list_has_category, dtype=torch.long)

    # # Weise die Edge-Indizes dem HeteroData-Objekt zu
    # data['paper', 'has_category', 'category'].edge_index = edge_index_tensor_has_category

    # %%
    # edge_index_tensor_has_category

    # %%
    # # Journalnamen in Nummern umwandeln
    # journal_to_index_in_journal = {journal: idx for idx, journal in enumerate(set(list_of_journals))}
    # edge_index_list_in_journal = [
    #     [journal_to_index_in_journal[journal] for journal in list_of_journals]
    # ]

    # # Konvertiere die Liste in torch.Tensor-Objekte
    # edge_index_tensor_in_journal = torch.tensor(edge_index_list_in_journal, dtype=torch.long)

    # # Weise die Edge-Indizes dem HeteroData-Objekt zu
    # data['paper', 'in_journal', 'journal-ref'].edge_index = edge_index_tensor_in_journal


    # %%
    # # Wörter in Nummern umwandeln
    # word_to_index_has_word = {word: idx for idx, word in enumerate(set(list_of_words))}
    # edge_index_list_has_word = [
    #     [word_to_index_has_word[word] for word in list_of_words]
    # ]

    # # Konvertiere die Liste in torch.Tensor-Objekte
    # edge_index_tensor_has_word = torch.tensor(edge_index_list_has_word, dtype=torch.long)

    # # Weise die Edge-Indizes dem HeteroData-Objekt zu
    # data['paper', 'has_word', 'word'].edge_index = edge_index_tensor_has_word


    # %%
    # # word in number
    # word_to_index = {word: idx for idx, word in enumerate(set(word for sublist in edge_index_list_has_titleword for word in sublist))}


    # # Erstelle die Edge-Index-Listen mit den gemappten Zahlen
    # edge_index_list_has_titleword = [
    #     [word_to_index[word] for word in sublist] for sublist in edge_index_list_has_titleword
    # ]

    # # Konvertiere die Listen in torch.Tensor-Objekte
    # edge_index_tensor_has_titleword = torch.tensor(edge_index_list_has_titleword, dtype=torch.long)

    # # Weise die Edge-Indizes dem HeteroData-Objekt zu
    # data['paper', 'has_titleword', 'word'].edge_index = edge_index_tensor_has_titleword

    # %%
    # data['paper', 'has_titleword', 'word'].edge_index

    # %% [markdown]
    # ## Create Hetero Object

    # %%


    data = HeteroData()
    data['paper'].num_nodes = len(df)
    data['paper'].license = licenses_list
    data['paper'].doi = doi_list
    data['paper'].title = title_list
    data['paper'].pages = pages_list
    data['paper'].journal = journal_list
    data['paper'].date = date_list
    data['paper'].id = id_list

    data['author'].num_nodes = len(authors_list)
    data['author'].name = authors_list

    data['category'].num_nodes = len(categories_list)
    data['category'].name = categories_list

    data['journal'].num_nodes = len(df)
    data['journal'].name = journal_list

    data['word'].num_nodes = len(words_list)
    data['word'].name = words_list
    torch.save(data, "hetero_graph_temp.pt")

else:
    data = torch.load("hetero_graph_temp.pt")

print('heterograph')
# %% [markdown]
# ### Create edges
if not os.path.exists("hetero_graph_temp3.pkl"):
    # %%
    edge_index_list_written_by = []
    list_of_authors = []
    list_of_titles = []
    print('x15')

    # Iteriere durch die Zeilen des DataFrames
    for i in tqdm(range(len(df))):
        # Holen der ID des Papers aus der aktuellen Zeile
        paper_title = df['title'][i]
        
        # Holen der Autoreninformationen aus der aktuellen Zeile
        authors_parsed = df['authors_parsed'][i]
        
        for j in authors_parsed:
            list_of_authors.append(j)
            list_of_titles.append(paper_title)

    edge_index_list_written_by.append(list_of_titles)
    edge_index_list_written_by.append(list_of_authors)


    # Konvertiere die Liste in ein torch.Tensor-Objekt
    # edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t()

    # Weise die Edge-Indizes dem HeteroData-Objekt zu
    data['paper', 'written_by', 'author'].edge_index = edge_index_list_written_by

    # %%
    # Autor in Zahl umwandeln
    print('author to index')
    author_to_index = {author: idx for idx, author in enumerate(set(author for sublist in edge_index_list_written_by for author in sublist))}

    # Erstelle die Edge-Index-Listen mit den gemappten Zahlen für die "written_by"-Kante
    edge_index_list_written_by = [
        [author_to_index[author] for author in sublist] for sublist in edge_index_list_written_by
    ]

    # Konvertiere die Listen in torch.Tensor-Objekte für die "written_by"-Kante
    edge_index_tensor_written_by = torch.tensor(edge_index_list_written_by, dtype=torch.long)

    # Weise die Edge-Indizes für die "written_by"-Kante dem HeteroData-Objekt zu
    data['paper', 'written_by', 'author'].edge_index = edge_index_tensor_written_by


    # %%
    edge_index_list_has_category = []
    list_of_paper_cat = []
    list_of_categories = []

    print('make categories')
    # Iteriere durch die Zeilen des DataFrames
    print('x16')

    for i in tqdm(range(len(df))):
        # Holen der ID des Papers aus der aktuellen Zeile
        paper_title = df['title'][i]
        
        # Holen der Kategorieninformationen aus der aktuellen Zeile
        categories = df['categories'][i]
        
        # Hier gehe ich davon aus, dass die Kategorien als Liste vorliegen
        for category in categories:
            list_of_categories.append(category)
            list_of_paper_cat.append(paper_title)

    edge_index_list_has_category.append(list_of_paper_cat)
    edge_index_list_has_category.append(list_of_categories)

    # Weise die Edge-Indizes dem HeteroData-Objekt zu
    data['paper', 'has_category', 'category'].edge_index = edge_index_list_has_category


    # %%
    print('category to index')
    # Kategorie in Zahl umwandeln
    category_to_index = {category: idx for idx, category in enumerate(set(category for sublist in edge_index_list_has_category for category in sublist))}

    # Erstelle die Edge-Index-Listen mit den gemappten Zahlen für die "has_category"-Kante
    edge_index_list_has_category = [
        [category_to_index[category] for category in sublist] for sublist in edge_index_list_has_category
    ]

    # Konvertiere die Listen in torch.Tensor-Objekte für die "has_category"-Kante
    edge_index_tensor_has_category = torch.tensor(edge_index_list_has_category, dtype=torch.long)

    # Weise die Edge-Indizes für die "has_category"-Kante dem HeteroData-Objekt zu
    data['paper', 'has_category', 'category'].edge_index = edge_index_tensor_has_category


    # %%
    category_to_index

    # %%
    edge_index_tensor_has_category

    # %%
    edge_index_list_has_word = []
    list_of_paper_word = []
    list_of_words = []
    import gc
    gc.collect()

    print('paper title indices')
    print('x17')

    # Iteriere durch die Zeilen des DataFrames
    for i in tqdm(range(len(df))):
        # Holen der ID des Papers aus der aktuellen Zeile
        paper_title = df['title'][i]
        
        # Holen der Wortinformationen aus der aktuellen Zeile
        words = all_words[i]
        
        # Hier gehe ich davon aus, dass die Wörter als Liste vorliegen
        for word in words:
            list_of_words.append(word)
            list_of_paper_word.append(paper_title)

    edge_index_list_has_word.append(list_of_paper_word)
    edge_index_list_has_word.append(list_of_words)

    # Weise die Edge-Indizes dem HeteroData-Objekt zu
    data['paper', 'has_word', 'word'].edge_index = edge_index_list_has_word

    # %%
    print('word to index')
    # Wort in Zahl umwandeln
    word_to_index = {word: idx for idx, word in enumerate(set(word for sublist in edge_index_list_has_word for word in sublist))}

    # Erstelle die Edge-Index-Listen mit den gemappten Zahlen für die "has_word"-Kante
    edge_index_list_has_word = [
        [word_to_index[word] for word in sublist] for sublist in edge_index_list_has_word
    ]

    # Konvertiere die Listen in torch.Tensor-Objekte für die "has_word"-Kante
    edge_index_tensor_has_word = torch.tensor(edge_index_list_has_word, dtype=torch.long)

    # Weise die Edge-Indizes für die "has_word"-Kante dem HeteroData-Objekt zu
    data['paper', 'has_word', 'word'].edge_index = edge_index_tensor_has_word

    # %%
    edge_index_list_has_titleword = []
    list_of_paper_titleword = []
    list_of_titlewords = []
    print('x18')

    # Iteriere durch die Zeilen des DataFrames
    for i in tqdm(range(len(df))):
        # Holen der ID des Papers aus der aktuellen Zeile
        paper_title = df['title'][i]
        
        # Holen der Wortinformationen aus der aktuellen Zeile (hier nehmen wir die Titelwörter)
        title_words = all_title_words[i]
        
        # Hier gehe ich davon aus, dass die Titelwörter als Liste vorliegen
        for title_word in title_words:
            list_of_titlewords.append(title_word)
            list_of_paper_titleword.append(paper_title)

    edge_index_list_has_titleword.append(list_of_paper_titleword)
    edge_index_list_has_titleword.append(list_of_titlewords)

    print('make paper words')
    # Weise die Edge-Indizes dem HeteroData-Objekt zu
    data['paper', 'has_titleword', 'word'].edge_index = edge_index_list_has_titleword

    # %%
    # word in number
    titleword_to_index = {word: idx for idx, word in enumerate(set(word for sublist in edge_index_list_has_titleword for word in sublist))}


    # Erstelle die Edge-Index-Listen mit den gemappten Zahlen
    edge_index_list_has_titleword = [
        [titleword_to_index[word] for word in sublist] for sublist in edge_index_list_has_titleword
    ]

    # Konvertiere die Listen in torch.Tensor-Objekte
    edge_index_tensor_has_titleword = torch.tensor(edge_index_list_has_titleword, dtype=torch.long)

    # Weise die Edge-Indizes dem HeteroData-Objekt zu
    data['paper', 'has_titleword', 'word'].edge_index = edge_index_tensor_has_titleword

    # %%
    edge_index_list_in_journal = []
    list_of_paper_journal = []
    list_of_journals = []

    print('make paper journal')
    print('x19')

    # Iteriere durch die Zeilen des DataFrames
    for i in tqdm(range(len(df))):
        # Holen der ID des Papers aus der aktuellen Zeile
        paper_title = df['title'][i]
        
        # Holen des Journalnamens aus der aktuellen Zeile
        journal = df['journal-ref'][i]
        
        # Überprüfe, ob der Journal-Eintrag NaN ist
        if pd.notna(journal):
            list_of_journals.append(journal)
            list_of_paper_journal.append(paper_title)

    # Füge nur gültige Einträge hinzu
    edge_index_list_in_journal.append(list_of_paper_journal)
    edge_index_list_in_journal.append(list_of_journals)

    # Weise die Edge-Indizes dem HeteroData-Objekt zu
    data['paper', 'in_journal', 'journal-ref'].edge_index = edge_index_list_in_journal

    # %%
    print('journal to index')
    journal_to_index = {journal: idx for idx, journal in enumerate(set(journal for sublist in edge_index_list_in_journal for journal in sublist))}

    # Erstelle die Edge-Index-Listen mit den gemappten Zahlen
    edge_index_list_in_journal = [
        [journal_to_index[journal] for journal in sublist] for sublist in edge_index_list_in_journal
    ]

    # Konvertiere die Listen in torch.Tensor-Objekte
    edge_index_tensor_in_journal = torch.tensor(edge_index_list_in_journal, dtype=torch.long)

    # Weise die Edge-Indizes dem HeteroData-Objekt zu
    data['paper', 'in_journal', 'journal-ref'].edge_index = edge_index_tensor_in_journal

    #torch.save(data, "hetero_graph_temp2.pt")
    with open('hetero_graph_temp3.pkl', 'wb') as f:
        pickle.dump(data, f)
else:
    print('skip')
    with open('hetero_graph_temp3.pkl', 'rb') as f:
        data = pickle.load(f)
        
corpus = [" ".join(words) for words in filtered_all_words]
if not os.path.exists("hetero_graph_tfidf.pkl"):
    
    # %%
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch
    print('tfidf')
    # TF-IDF Vektorizer
    vectorizer = TfidfVectorizer()
    
    print('tfidfvector')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print('tfidf_vector finished')
    # Erstellen eines Wörterbuches für Feature-Namen
    feature_names = vectorizer.get_feature_names_out()
    print('got names out')
    word_to_index = {word: idx for idx, word in enumerate(feature_names)}

    # Initialisierung der Listen
    list_of_paper_word = []
    list_of_words = []
    tfidf_weights = []
    print('x20')

    # Extraktion der TF-IDF-Werte
    for i in tqdm(range(len(df))):
        paper_title = df['title'][i]
        unique_words = set(filtered_all_words[i])  # Verwenden eines Sets zur Reduzierung von Duplikaten

        for word in unique_words:
            if word in word_to_index:
                word_index = word_to_index[word]
                tfidf_weight = tfidf_matrix[i, word_index]
                if tfidf_weight > 0:  # Fügen Sie nur Kanten für positive TF-IDF-Werte hinzu
                    list_of_paper_word.append(paper_title)
                    list_of_words.append(word)
                    tfidf_weights.append(tfidf_weight)

    print('x21')
    print('list of paper word',len(list_of_paper_word))
    print('list of words',len(list_of_words))
    # Erstellen der Kanten für das HeteroData-Objekt
    # edge_index_list_has_word = [[], []]
    # for i in tqdm(range(len(list_of_paper_word))):
    #     paper_title, word = list_of_paper_word[i], list_of_words[i]
    #     # Finde den Index des Papiers basierend auf dem Titel
    #     paper_idx = df.index[df['title'] == paper_title].tolist()
    #     if paper_idx:  # Sicherstellen, dass ein Index gefunden wurde
    #         paper_idx = paper_idx[0]
    #         word_idx = word_to_index[word]
    #         edge_index_list_has_word[0].append(paper_idx)
    #         edge_index_list_has_word[1].append(word_idx)

    paper_title_indices = {paper_title: idx for idx, paper_title in enumerate(df['title'])}
    temp_word_to_index = [word_to_index[word] for word in list_of_words]
    temp_paper_title_indices = [paper_title_indices[paper_title] for paper_title in list_of_paper_word]
    edge_index_list_has_word = [temp_paper_title_indices, temp_word_to_index]

    data['paper', 'has_word', 'word'].edge_index = torch.tensor(edge_index_list_has_word, dtype=torch.long)
    data['paper', 'has_word', 'word'].edge_attr = torch.tensor(tfidf_weights, dtype=torch.float)


    with open('hetero_graph_tfidf.pkl', 'wb') as f:
        pickle.dump(data, f)
        
else:
    print('skiptfidf')
    with open('hetero_graph_tfidf.pkl', 'rb') as f:
        data = pickle.load(f)
# %%


# %%
# Erstellen eines DataFrame
# tfidf_df = pd.DataFrame({
#     'paper_title': list_of_paper_word,
#     'word': list_of_words,
#     'tfidf_weight': tfidf_weights
# })
# tfidf_df.tfidf_weight.hist(bins=100)

# %%
# from sklearn.feature_extraction.text import TfidfVectorizer
# from collections import Counter
# import math

# # TF-IDF Vektorizer
# vectorizer = TfidfVectorizer()
# corpus = [" ".join(words) for words in filtered_all_words]
# tfidf_matrix = vectorizer.fit_transform(corpus)

# feature_names = vectorizer.get_feature_names_out()
# feature_names_list = feature_names.tolist()

# edge_index_list_has_word, list_of_paper_word, list_of_words, tfidf_weights  = [], [], [], []

# for i in tqdm(range(len(df)):
#     paper_title = df['title'][i]
#     words = filtered_all_words[i]

#     for word in words:
#         if word in feature_names:  # Prüfen, ob das Wort im Vektorizer vorhanden ist, Bindewörter nicht mehr drinnen
            
#             word_index = feature_names_list.index(word)
#             tfidf_weight = tfidf_matrix[i, word_index]

#             list_of_paper_word.append(paper_title)
#             list_of_words.append(word)
#             tfidf_weights.append(tfidf_weight)

# # Zuweisung der Kanten und Gewichte
# edge_index_list_has_word.append(list_of_words)
# data['paper', 'has_word', 'word'].edge_index = edge_index_list_has_word
# data['paper', 'has_word', 'word'].edge_attr = torch.tensor(tfidf_weights, dtype=torch.float)

# %%
# # Erstellen eines DataFrame
# tfidf_df = pd.DataFrame({
#     'paper_title': list_of_paper_word,
#     'word': list_of_words,
#     'tfidf_weight': tfidf_weights
# })

# %%
# tfidf_df.tfidf_weight.hist(bins=100)

# %% [markdown]
# ##### tf-idf short version

# %%
# https://medium.com/analytics-vidhya/demonstrating-calculation-of-tf-idf-from-sklearn-4f9526e7e78b

# %%
# # short way:

# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer()
# corpus = [" ".join(words) for words in filtered_all_words]
# tfidf_matrix = vectorizer.fit_transform(corpus)

# feature_names = vectorizer.get_feature_names_out()

# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# corpus

# %%
# tfidf_df

# %% [markdown]
# ##### step by step to check values

# %%
# https://medium.com/analytics-vidhya/demonstrating-calculation-of-tf-idf-from-sklearn-4f9526e7e78b

# %%
# from sklearn.feature_extraction.text import CountVectorizer
# import pandas as pd

# # Erstellen eines CountVectorizer-Objekts
# count_vectorizer = CountVectorizer()
# corpus = [" ".join(words) for words in filtered_all_words]

# # Berechnen der Term Frequency Matrix
# count_matrix = count_vectorizer.fit_transform(corpus)

# words_per_document = count_matrix.sum(axis=1).A1
# counter_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
# counter_df

# %%
# # Normalisieren der Zählungen, um TF-Werte zu erhalten
# tf_matrix = count_matrix / words_per_document[:, None]

# # Umwandeln in ein DataFrame für eine bessere Darstellung (optional)
# tf_df = pd.DataFrame(tf_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
# tf_df

# %%
# from sklearn.feature_extraction.text import TfidfTransformer

# tfidf_transformer = TfidfTransformer()
# X = tfidf_transformer.fit_transform(count_matrix)
# idf = pd.DataFrame({'feature_name':count_vectorizer.get_feature_names_out(), 'idf_weights':tfidf_transformer.idf_})
# idf

# %%
# idf_series = idf.set_index('feature_name')['idf_weights']
# idf_dict = idf_series.to_dict()

# # Erstellen einer Kopie des TF-DataFrames
# tfidf = counter_df.copy()

# # Multiplizieren der TF-Werte mit den entsprechenden IDF-Werten
# for col in tfidf.columns:
#     if col in idf_dict:
#         tfidf[col] = tfidf[col] * idf_dict[col]

# tfidf

# %%
# import numpy as np

# sqrt_vec = np.sqrt(tfidf.pow(2).sum(axis=1))
# tfidf.div(sqrt_vec, axis=0)

# %% [markdown]
# #### PMI
print('pmi')
# %%
# https://www.listendata.com/2022/06/pointwise-mutual-information-pmi.html

# %%
import torch
from torch_geometric.data import HeteroData
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import nltk
nltk.download('punkt')
# Vorbereiten des Korpus und Finden der Bigramme

full_text = ' '.join(corpus)
tokens = word_tokenize(full_text)

bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens)


# Sammeln aller einzigartigen Wörter
unique_words = set(word for bigram in finder.ngram_fd for word in bigram)

# Erstellen eines Mappings von Wörtern zu Indizes
word_to_index = {word: i for i, word in enumerate(unique_words)}

# Initialisieren der Kanten und PMI-Werte
edge_index = [[], []]
pmi_values = []

# Zuweisen der PMI-Werte als Kantenattribute
for bigram, pmi_score in tqdm(finder.score_ngrams(bigram_measures.pmi)):
    if pmi_score > 0:
        word1, word2 = bigram
        edge_index[0].append(word_to_index[word1])
        edge_index[1].append(word_to_index[word2])
        pmi_values.append(pmi_score)

# Konvertieren in PyTorch Tensoren
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
pmi_values_tensor = torch.tensor(pmi_values, dtype=torch.float)

# Hinzufügen der Kanten und Kantenattribute zum HeteroData-Objekt
edge_type = ('word', 'co_occurs_with', 'word')

data[edge_type].edge_index = edge_index_tensor
data[edge_type].edge_attr = pmi_values_tensor


# %%
# plt.hist(data[edge_type].edge_attr.numpy(), bins=100, edgecolor='black')
# plt.xlabel('Wert')
# plt.ylabel('Häufigkeit')
# plt.title('Histogramm der Tensor-Werte')
# plt.show()

# %%
# data.edge_attrs

# %%
# print(data.node_types)
# print(data.edge_types)

# %% [markdown]
# ## Scale weights for training??

# %% [markdown]
# # Export

# %%
#import pickle
print("save final")
# Serialize and save the hetero-object
with open('hetero_graph_final.pkl', 'wb') as f:
    pickle.dump(data, f)

# %%
# Load the hetero-object
with open('hetero_graph_final.pkl', 'rb') as f:
    loaded_hetero_graph = pickle.load(f)
    
loaded_hetero_graph["author"]

# %% [markdown]
# # Visualization

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
from math import log
import torch
import networkx as nx
import matplotlib.pyplot as plt

# %%
from sklearn.feature_extraction.text import CountVectorizer

# Ihr DataFrame und die Spalte 'abstract'
abstracts = df['abstract'].tolist()

counter = CountVectorizer(stop_words='english')
count_matrix = counter.fit_transform(abstracts)

# Verwenden Sie die entsprechende Methode je nach Ihrer scikit-learn Version
try:
    features = counter.get_feature_names_out()
except AttributeError:
    features = counter.get_feature_names()

abstracts_counter = pd.Series(count_matrix.toarray().sum(axis=0), 
                              index=features).sort_values(ascending=False)

bar_graph = abstracts_counter[:50].plot(kind='bar', figsize=(18,8), alpha=1, fontsize=17, rot=90, edgecolor='black', linewidth=2,
            title='Word Counts in Abstracts')
bar_graph.set_xlabel('Words')
bar_graph.set_ylabel('Occurrences')
bar_graph.title.set_size(18)
plt.show()


# %%
# import pandas as pd
# import numpy as np
# import networkx as nx
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import Counter
# import math
# import itertools

# # Angenommen, df ist Ihr DataFrame mit einer Spalte 'document' für Dokumente
# documents = df['title'][:1].tolist()

# # Berechnung der TF-IDF-Werte
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
# feature_names = tfidf_vectorizer.get_feature_names_out()

# # Erstellung des Graphen
# G = nx.Graph()

# # Hinzufügen von Dokument- und Wortknoten
# for doc_id, doc in enumerate(documents):
#     G.add_node(f'doc_{doc_id}', type='document')

# for word in feature_names:
#     G.add_node(word, type='word')

# # Hinzufügen von Dokument-Wort-Kanten mit TF-IDF-Gewichtungen
# for doc_id in tqdm(range(tfidf_matrix.shape[0]):
#     for word_id in tfidf_matrix[doc_id].nonzero()[1]:
#         tfidf_weight = tfidf_matrix[doc_id, word_id]
#         G.add_edge(f'doc_{doc_id}', feature_names[word_id], weight=tfidf_weight)
#         print(f"doc_id: {doc_id}, word_id: {feature_names[word_id]}, weight: {tfidf_weight}")
        
# # Berechnung der PMI-Werte für Wort-Wort-Kanten
# word_counts = Counter(np.array(tfidf_matrix.sum(axis=0)).flatten())
# total_count = sum(word_counts.values())
# word_pairs = itertools.combinations(feature_names, 2)

# for word1, word2 in word_pairs:
#     count_word1 = word_counts[word1]
#     count_word2 = word_counts[word2]
#     count_word1_word2 = (tfidf_matrix[:, feature_names.tolist().index(word1)] 
#                          + tfidf_matrix[:, feature_names.tolist().index(word2)]).sum()
#     pmi = math.log((count_word1_word2 * total_count) / (count_word1 * count_word2), 2)
#     if pmi > 0:
#         G.add_edge(word1, word2, weight=pmi)

# # Zeige den Graphen an, deaktiviere die Verarbeitung von LaTeX-Symbolen
# plt.rcParams['text.usetex'] = False

# # circular_layout sonst Zahl zu groß für Visualisierung von pmi, davor spring genommen
# pos = nx.circular_layout(G)  # Layout des Graphen
# nx.draw(G, pos, with_labels=True, font_size=6, node_size=100, node_color='lightblue', font_color='black', font_weight='bold', edge_color='gray')

# # # Gewichte der Kanten holen
# # edge_weights = nx.get_edge_attributes(G, 'weight')
# # # Kanten-Labels zeichnen
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=5)

# plt.title("Graph Representation of the ArXiv Dataset")
# plt.show()


