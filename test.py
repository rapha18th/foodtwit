import pyTigerGraph as tg # Importing
import pandas as pd
import json


#for i in countries:
 #     print(str(i)+" food shortage")


# Create a Connection
conn = tg.TigerGraphConnection(host="https://foodtwit.i.tgcloud.io", username="tigergraph",
                               password="tigergraph")

print(conn.gsql('ls', options=[]))

conn.graphname = "FOODNEWS3"
secret = conn.createSecret()
authToken = conn.getToken(secret)
authToken = authToken[0]
print(authToken)

conn.graphname = "FOODNEWS3"
secret = conn.createSecret()
authToken = conn.getToken(secret)
authToken = authToken[0]
print(authToken)

articles_file = 'articles.csv'
results = conn.uploadFile(articles_file, fileTag='MyDataSource', jobName='load_articles')
print(json.dumps(results, indent=2))

topics_file = 'topics.csv'
results=conn.uploadFile(topics_file, fileTag='MyDataSource', jobName='load_topics')
print(json.dumps(results, indent=2))

s_file= 'sentiment.csv'
results=conn.uploadFile(s_file, fileTag='MyDataSource', jobName='load_sentiment')
print(json.dumps(results, indent=2))

l_file = 'labels.csv'
results=conn.uploadFile(l_file, fileTag='MyDataSource', jobName='load_labels')
print(json.dumps(results, indent=2))
