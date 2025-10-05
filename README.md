# Loanword detection in Armenian

In each language there are words which are loaned from other languages or shared between them. This work is an attempt to learn the patterns among such words in Armenian and predict their original languages.

## Raw data

This loanword data (862 items) was collected manually from the Armenian dictionary Nayiri during several months; see the processed data in the file `data/loanword_data.json`. These are words across 36 languages.

The native Armenian word data (865 items) was borrowed from [Wikipedia](https://hy.wikipedia.org/wiki/%D4%B2%D5%B6%D5%AB%D5%AF_%D5%B0%D5%A1%D5%B5%D5%A5%D6%80%D5%A5%D5%B6_%D5%A2%D5%A1%D5%BC%D5%A5%D6%80); see the processed data in the file `data/armenian_data.json`.

For training were used only 788 words from the most prevalent  languages (excluding native Armenians).

## Data processing

1. All the words were split into syllables by hand. These syllables were used in training to deliver the unique language patterns. They were vectorized by scikit-learn's `CountVectorizer`.

2. Character level features, and more broadly $n$-grams (for $n \in \{1, 2, 3\}$), were used too. They were also vectorized by `CountVectorizer`.

3. Byte-pair enoding (BPE) was utilized as well, by Hugging Face's `tokenizers`. Its features were transformed by scikit-learn's `TfidfVectorizer`.

4. Because of the high number of features scikit-learn's `TruncatedSVD` was used to reduce data's dimensionality.

## Applied models

Three different classic ML models were applied: scikit-learn's `LogisticRegression`, `RandomForestClassifier` and CatBoost's `CatBoostClassifier`.

Since each word could be coming from several languages, this is a multilabel classification task. Thus scikit-learn's `OneVsRestClassifier` was used to wrap each of the above three models.

## Results

| Model                 | Features                              | F1-score |
|-----------------------|---------------------------------------|----------|
| LogisticRegression    |syllables                              | 0.571    |
| LogisticRegression    |syllables + ngrams                     | 0.652    |
| LogisticRegression    |syllables + ngrams + BPE               | 0.656    |
| RandomForestClassifier|syllables + ngrams + BPE               | 0.669    |
| CatBoost              |syllables + ngrams + BPE + TruncatedSVD| 0.56     |

One can see that `RandomForestClassifier` outperforms the other models with its 0.67 F1-score.

## Comments

1. Due to data scarcity deep learning approaches are not practical.

1. The relatively low metrics above can also be justified by fewness of data. Another reason is the sharp class imbalance.

1. The native Armenian words contained verbs which added unfair bias because of verb endings. It is worth to do try the above approaches on the dataset without verbs as well.

1. Syllables are "armanianized", meaning that they may not fully reflect the original language's patterns.

## TODO

1. Remove verbs.

1. Try clusterization.

1. Train on the updated dataset.

1. Evaluate on unlabeled words as an experiment.

1. Perform an ablation study.
