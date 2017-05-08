# coding=utf-8
#!/usr/bin/env python


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from read_data import read_reviews

categories = ['neg', 'pos']
base_path = '/Users/Neyanbhbin/Documents/code/cs585/project-angelok/aclImdb'

data, labels = read_reviews(base_path, data_type='train', labels=categories)
test_data, test_labels = read_reviews(base_path, data_type='test', labels=categories)

data_all = data + test_data

def imdb_vocabulary(remove_stopwords=True, min_df=3, use_idf=1, ngram_range=(1,1)):
    stop_words = 'english' if remove_stopwords else ''

    count_vectorizer = TfidfVectorizer(min_df=min_df, use_idf=use_idf,
                                        ngram_range=ngram_range)
    X = count_vectorizer.fit_transform(data_all)
    print(X.shape)
    vo_list = sorted(count_vectorizer.vocabulary_.keys())
    vo_list = [s for s in vo_list if isinstance(s, str)] 
    content = "\n".join(vo_list)
    with open("vocabulary.txt", "w") as f:
        f.write(content)

if __name__ == "__main__":
    imdb_vocabulary()

