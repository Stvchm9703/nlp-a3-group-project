## Popular NER Datasets

1. **CoNLL-2003 (English)**
   - **Source**: Conference on Computational Natural Language Learning (CoNLL) 2003.
   - **Entities**: PERSON, LOCATION, ORGANIZATION, MISC.
   - **Language**: English.
   - **Details**: Consists of news wire articles from the Reuters Corpus, widely used for benchmarking NER systems.

2. **OntoNotes 5.0**
   - **Source**: A collaborative effort among several research groups.
   - **Entities**: Includes a wide range such as PERSON, LOCATION, ORGANIZATION, and others.
   - **Languages**: English, Chinese, Arabic.
   - **Details**: Combines data from various domains like news, conversational telephone speech, weblogs, and more, providing rich annotations including syntactic and semantic information.

3. **ACE (Automatic Content Extraction)**
   - **Source**: Competitions organized by NIST.
   - **Entities**: PERSON, ORGANIZATION, LOCATION, GPE, and others.
   - **Languages**: Multiple languages including English, Arabic, and Chinese.
   - **Details**: Aims to develop capabilities for automatic processing of human languages, with annotations for entities, relations, and events.

4. **MUC (Message Understanding Conference)**
   - **Source**: Early NER competitions in the 1990s.
   - **Entities**: PERSON, ORGANIZATION, LOCATION, TIME, and others.
   - **Language**: Primarily English.
   - **Details**: One of the first efforts to formalize NER tasks, creating large-scale annotations used for training early NER systems.

5. **Wikigold**
   - **Source**: Created from Wikipedia articles as a gold standard dataset.
   - **Entities**: Similar to CoNLL-2003.
   - **Language**: English.
   - **Details**: Useful for additional testing beyond the training data provided by more standard datasets.

##  CoNLL-2003 Datasets

1. **CoNLL-2003 (English)**
   - **Size**: The dataset contains a total of 14,987 sentences, divided into 3,684 sentences for training, 3,466 for validation, and 3,684 for testing.
   - **Data Points**: Includes over 300,000 tokens annotated with named entity tags.
   - **Usage Decision**: Decided to use this dataset for upcoming projects due to its comprehensiveness and established benchmarking capabilities.

## GloVe: Global Vectors for Word Representations
curl -O https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip

