FROM omairs/foodome2_no_data

RUN apt-get update 
RUN apt-get install unzip

RUN mkdir /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADeWMv-MB1n-dAHRG6z3I-Oa/sars-busters/Mol2vec/amino_01_w_embed.pkl -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABN1FjfiQXWoRJX4rcNdhHCa/sars-busters/Mol2vec/chemicals_01_w_embed.pkl -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADhCbPEodDjLo3YqILPgDHGa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_edges_0.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAB3xHWOrADedOBX6_53zxuZa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_edges_1.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAAcxVZLf8e6d1rXeonOI2CIa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_edges_2.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACf_6zR0XZVn42o8StEgxk-a/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_edges_3.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAD0hFSU_L5VZMDAch9JHDO3a/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_edges_4.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAAFPoqsUc1P-e48XTlZ0u4Ea/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_nodes_0.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABW8NkCfEr-QpUC4NNNW07Oa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_nodes_1.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADp7xTeWnNnjzP2zdxPASpQa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_nodes_2.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABgI2vDq8gKqls6VF-pxtX4a/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_nodes_3.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABOrp-wb-gQrj7iQOOFYU5Da/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/test_unseen_nodes_4.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACvX1duWQbF_hiM-lczNdG0a/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_edges_0.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAC6UNXMqZDyM9_rkN6fRC9Ja/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_edges_1.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADllne-nPrs3Yp0R0fSO7Bia/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_edges_2.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACzAKXi0gNWAE23i4MmNkWZa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_edges_3.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACEZ4hmU3kIUwWhNCpVHbIna/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_edges_4.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADX1blOezgORbhiAcfWMTnKa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_nodes_0.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAAXcJYvtcNtqwJwN5Q5q0sUa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_nodes_1.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADf9ewosrd9_IaFxE4EqqJta/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_nodes_2.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACr64pjLYFBnzoCYhJz5LKsa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_nodes_3.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACnZrcTUsuxUtLrNmUClz6Ua/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/validation_unseen_nodes_4.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABqWHQF6mqaQK5Th1PHZUQia/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/train_0.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACIGFJLsOcyStBZpQN-hdDDa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/train_1.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAAq0u4wVEC6Tm6ydiG7JKHma/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/train_2.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABR3m1bpvm6Ne92FOTKdbfNa/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/train_3.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABFx6GR0nPx44Po6EPXE99_a/sars-busters-consolidated/GitData/VecNet_Unseen_Nodes/train_4.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADyQIzgygAjcI0BYkG-PtWSa/sars-busters-consolidated/GitData/interactions/Network_Derived_Negatives.csv -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AAB6n4hxT61HXCJ3zzVTAkVYa/sars-busters-consolidated/GitData/VecNet_unseen_nodes.pickle -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AACO-KxBWgTdEIouMhzDEDBca/sars-busters/Mol2vec/model_300dim.pkl -P /root/data \
&& wget https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABUBQpaLezSGHGIrPtEPwlma/sars-busters/Mol2vec/results/protVec_100d_3grams.csv -P /root/data \
&& wget --max-redirect=20 -O /root/data/vecnet-final.zip https://www.dropbox.com/sh/i2gixtsik1qbjxq/AABCwJylHLsBWFWpWkWrNnfya/sars-busters-consolidated/GitData/vecnet-final \
&& unzip /root/data/vecnet-final.zip -d ./vecnet-final 
