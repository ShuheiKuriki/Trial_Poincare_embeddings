# -*- coding: utf-8 -*-

train_data = [
    ('Software Engineer', 'Engineer'),
    ('Senior Software Engineer', 'Software Engineer'),
    ('Web Programmer', 'Programmer'),
    ('Programmer', 'エンジニア'),
    ('UI Designer', 'Designer'),
    ('Designer', 'Engineer'),
    ('エンジニア', 'Engineer'),
    ('Engineer', 'エンジニア')
]

labels = []
for u,v in train_data:
    if u not in labels:
        labels.append(u)
    if v not in labels:
        labels.append(v)

from gensim.models.poincare import PoincareModel
from gensim.viz.poincare import poincare_2d_visualization
from IPython import display
from plotly.offline import plot
import pandas as pd



# import csv

# # ファイルオープン
# f = open('output.csv', 'w')
# writer = csv.writer(f, lineterminator='\n')

# # データをリストに保持
# csvlists = [
#     ['Software Engineer', 'Engineer'],
#     ['Senior Software Engineer', 'Software Engineer'],
#     ['Web Programmer', 'Programmer'],
#     ['UI Designer', 'Designer'],
#     ['エンジニア', 'Engineer'],
#     ['Engineer', 'エンジニア'],
# ]


# # 出力
# for csvlist in csvlists:
#     writer.writerow(csvlist)

# # ファイルクローズ
# f.close()

# occupation_relations_list = [(a, b) for a, b in pd.read_csv('output.csv', header=None).values]

# occupation_relations_list

from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
# relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))

model = PoincareModel(train_data, size=2, negative=5)
model.train(epochs=100)

# plotly.offline.init_notebook_mode(connected=False)
train_set = set(train_data)
figure_title = ""
plot(poincare_2d_visualization(model, train_set, figure_title, num_nodes=None, show_node_labels=labels))
