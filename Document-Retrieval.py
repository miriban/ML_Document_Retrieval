"""
    Document Retrieval
    Created By: Mohammed AbuIriban @mohammediriban
"""
import graphlab as gl

people = gl.SFrame("people_wiki.gl/")

people['word_count'] = gl.text_analytics.count_words(people['text'])

people['tfidf'] = gl.text_analytics.tf_idf(people['word_count'])

elton = people[people['name'] == 'Elton John']

knn_model = gl.nearest_neighbors.create(people,features=['tfidf'],label='name')

print knn_model.query(elton)
