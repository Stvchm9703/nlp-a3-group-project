# Features of the CONLL2003 Dataset

The dataset contains the following fields, which represent various linguistic annotations for each token in the sentences:

## `id`
- **Type**: `string`
- **Description**: A unique identifier for each example in the dataset.

## `tokens`
- **Type**: `Sequence`
- **Description**: A list of tokens (words) in a sentence.
- **Details**: Each token is a string.

## `pos_tags`
- **Type**: `Sequence`
- **Description**: Part-of-speech tags for each token in the `tokens` sequence.
- **Details**: 
  - Each POS tag is represented by a `ClassLabel`, which maps an integer to a string representing the POS tag. Some examples of POS tags include:
    - `NN`: Noun, singular or mass
    - `VB`: Verb, base form
    - `JJ`: Adjective
    - More specialized tags like `NNP` (proper noun, singular), `PRP$` (possessive pronoun), etc.
  - The full list of tags reflects the Penn Treebank POS tags, which are commonly used in NLP.

## `chunk_tags`
- **Type**: `Sequence`
- **Description**: Chunk tags represent syntactic constituents such as noun phrases or verb phrases.
- **Details**:
  - Each chunk tag is also a `ClassLabel`, similar to `pos_tags`.
  - Tags are in the IOB (Inside, Outside, Beginning) format, which is used to denote multi-token chunks.
  - Examples include:
    - `B-NP`: Beginning of a noun phrase
    - `I-NP`: Inside a noun phrase
    - `B-VP`: Beginning of a verb phrase
    - Other tags like `B-ADJP` (Adjective Phrase), `I-ADVP` (Adverb Phrase), etc.

## `ner_tags`
- **Type**: `Sequence`
- **Description**: Named Entity Recognition tags for each token in the `tokens` sequence.
- **Details**:
  - Each NER tag is a `ClassLabel`.
  - Tags include:
    - `O`: Outside any named entity
    - `B-PER`: Beginning of a person’s name
    - `I-PER`: Inside a person’s name
    - `B-ORG`: Beginning of an organization's name
    - `I-ORG`: Inside an organization's name
    - `B-LOC`: Beginning of a location name
    - `I-LOC`: Inside a location name
    - `B-MISC`: Beginning of a miscellaneous entity
    - `I-MISC`: Inside a miscellaneous entity
  - The B/I prefix indicates the Beginning/Inside part of an entity, useful for entities that span multiple tokens.

