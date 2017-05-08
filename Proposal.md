## Overview

I propose to analyze sentiment of texts by using multiple machine learning algorithms including logistic regression, SVM, naive bayes and neural networks and compare their performance. Sentiment analysis is an important problem of NLP and it's a good method to evaluate the quality of an e-commerce product. However, due to the large size of text data and the ambiguity of nature language, Accurately determining the type of a sequence of words is tough. In this project, I'll use the algorithms mentioned above to fit the IMDB movie reviews data and evaluate the performance.

## Data

The data have been collected from [IMDB](www.imdb.com) and are available at [Stanford IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

## Method

I'll use logistic regression, SVM, naive bayes and neural networks. They are provided by sklearn.

## Related Work

Emotion Classification Using Web Blog Corpora (http://ieeexplore.ieee.org/abstract/document/4427100/)

Integrating Labeled Latent Dirichlet Allocation into sentiment analysis of movie and general domains (http://ieeexplore.ieee.org.ezproxy.gl.iit.edu/document/7886071/)

Twitter sentiment analysis: The good the bad and the omg! (http://www.aaai.org/ocs/index.php/ICWSM/ICWSM11/paper/download/2857/3251?height%3D90%%26iframe%3Dtrue%26width%3D90%)

Comparison Research on Text Pre-processing Methods on Twitter Sentiment Analysis (http://ieeexplore.ieee.org/document/7862202/)

The Role of Text Pre-processing in Opinion Mining on a Social Media Language Dataset (http://ieeexplore.ieee.org/document/6984806/)

## Evaluation

The evaluation process contains results comparison and time consuming comparison. Results comparison contains accuracy rate and confusion matrix evaluation. Since the label of each review is known, we'll use the true label as the baseline. The key table is the confusion matrix table. There are key plots including prediction accuracy with different parameters and keyword cloud. 
