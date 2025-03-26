# lyric-information-retrieval
Information Retrieval for song lyrics 

## Motivation
It's a project for my NLP & IR class.

- I decided to apply information retrieval (IR) on songtexts of different genres and topics.
- Music is often analyzed by the instrumental melody, but a user might try to query for specific lyrics.
- It gives an interesting perspective on IR and I will see how good it works, because the language is natural human, but with a higher rate of metaphors and poetic elements instead of straightforward facts.
- The example collection of unstructured data contains 450 songs, primarily in English from the last 10 years and the pop genre.

## Description

The project is containing the following parts:

1. An example collection of unstructured data
1. An information retrieval system based on the Boolean Model	
1. An information retrieval system based on the Vector Space Model

### Information Retrieval
There are two systems: The Boolean Model and the Vector Space Model

#### Boolean Model

- Provides all songs that fullfill a users query
- Query is a boolean expression (example form: `term1 AND NOT term2 OR term3`)
- Information Retrieval based on set Theory with inverted index
- The inverted index is created by tokenizing the lyrics of the songs (form: `[(term1, [song1, song2, ...]), (term2, [song3, song4, ...]), ...]`)
- Result are the files of all songs that fullfill the query, can be found in the `output` folder

#### Vector Space Model

- Provides all songs that are most similar to a users query
- Query is a string of words (example form: `term1 term2 term3`)
- Information Retrieval based on the cosine similarity of the query and the songs
- Songs and Query are represented as vectors
- Formular used for Songs: (tf / max(tf)) * log2(N / df)
- Formular used for Query: (1 + log(tf)) * log2(N / df) -> Here might be possible improvements
- Result are: the files of all songs that are most similar to the query, a csv called "results_overview.csv" with the queries results in order

### Collection of Unstructured Data 

- csv of song names and their artists exported via [Exportify](https://exportify.net/#playlists)
- currently those are 1000 songs i know (It's mostly english songs with some single songs that are in other languages, that won't be a problem for information retrieval, because the songs just won't be considered as important)
- appr. 450 of the songs lyrics are retrieved with the [lyricsgenius library](https://pypi.org/project/lyricsgenius/)
- the code for this is not yet in this repo, i like keeping it to myself :)

## Start
- `.\env\Scripts\Activate `
- `python main.py`

