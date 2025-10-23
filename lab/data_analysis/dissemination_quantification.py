import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from hawkeslib.model import UnivariateExpHawkesProcess
import torch
import torch_geometric as pyg
from torch_geometric.nn import GraphConv
import networkx as nx
from transformers import BertTokenizer, BertForSequenceClassification
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

# 连接SQLite数据库
conn = sqlite3.connect('social_media.db')
cursor = conn.cursor()

# 1. 传播范围量化
def quantify_spread_range():
    # 从post表获取转发量数据
    query = "SELECT post_id, num_shares, created_at FROM post"
    df_posts = pd.read_sql_query(query, conn)
    
    # 负二项回归：预测转发量
    X = df_posts[['num_likes', 'num_dislikes']]  # 情绪强度因子
    y = df_posts['num_shares']
    X = sm.add_constant(X)
    nb_model = NegativeBinomial(y, X).fit()
    print("负二项回归结果：", nb_model.summary())
    
    # Hawkes模型：捕捉时间依赖性
    timestamps = pd.to_datetime(df_posts['created_at']).astype(np.int64) // 10**9
    hawkes = UnivariateExpHawkesProcess()
    hawkes.fit(timestamps.values)
    mu, delta = hawkes.mu, hawkes.delta
    print(f"Hawkes模型参数：基础传播率μ={mu:.2f}, 衰减系数δ={delta:.2f}")

# 2. 传播深度量化
def quantify_spread_depth():
    # 构建社交网络图（基于follow表）
    query = "SELECT follower_id, followee_id FROM follow"
    edges = pd.read_sql_query(query, conn)
    G = nx.DiGraph()
    G.add_edges_from(edges[['follower_id', 'followee_id']].values)
    
    # PageRank计算关键节点
    pagerank = nx.pagerank(G, alpha=0.85)  # 可通过网格搜索优化alpha
    print("PageRank前5节点：", sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # GraphConv模型学习传播路径
    edge_index = torch.tensor(edges[['follower_id', 'followee_id']].values.T, dtype=torch.long)
    x = torch.randn((G.number_of_nodes(), 16))  # 节点特征（可替换为实际特征）
    data = pyg.data.Data(x=x, edge_index=edge_index)
    model = GraphConv(in_channels=16, out_channels=8)
    out = model(data.x, data.edge_index)
    print("GraphConv嵌入向量维度：", out.shape)

# 3. 群体反应量化
def quantify_group_reaction():
    # 从comment表获取评论数据
    query = "SELECT comment_id, content FROM comment"
    df_comments = pd.read_sql_query(query, conn)
    
    # BERT情感分类
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    sentiments = []
    for text in df_comments['content']:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        sentiment = torch.argmax(outputs.logits, dim=1).item()
        sentiments.append(sentiment)
    df_comments['sentiment'] = sentiments
    print("情感分类结果：", df_comments['sentiment'].value_counts())
    
    # 贝叶斯网络建模用户决策
    query = "SELECT user_id, post_id, num_likes, num_dislikes FROM post"
    df_posts = pd.read_sql_query(query, conn)
    bn_model = BayesianNetwork([('num_likes', 'accept'), ('num_dislikes', 'accept')])
    bn_model.fit(df_posts, estimator=MaximumLikelihoodEstimator)
    print("贝叶斯网络结构：", bn_model.edges())

# 4. 干预效果评估（DID模型）
def evaluate_intervention():
    # 从post和report表获取干预组和对照组数据
    query = """
    SELECT p.post_id, p.num_shares, p.num_likes, p.created_at, r.report_reason
    FROM post p
    LEFT JOIN report r ON p.post_id = r.post_id
    """
    df = pd.read_sql_query(query, conn)
    
    # 构造面板数据
    df['time'] = pd.to_datetime(df['created_at']).dt.day
    df['treated'] = df['report_reason'].notnull().astype(int)  # 干预组：有report_reason
    df['post'] = 1  # 时间后处理标志（假设干预在某一时间点后）
    df['treated_post'] = df['treated'] * df['post']
    
    # PanelOLS模型
    df_panel = df.set_index(['post_id', 'time'])
    model = PanelOLS(df_panel['num_shares'], df_panel[['treated', 'post', 'treated_post', 'num_likes']], 
                     entity_effects=True, time_effects=True)
    results = model.fit()
    print("DID模型结果：", results.summary)
    
    # 可视化平行趋势
    plt.plot(df[df['treated'] == 0].groupby('time')['num_shares'].mean(), label='Control')
    plt.plot(df[df['treated'] == 1].groupby('time')['num_shares'].mean(), label='Treated')
    plt.legend()
    plt.show()

# 执行分析
if __name__ == "__main__":
    quantify_spread_range()
    quantify_spread_depth()
    quantify_group_reaction()
    evaluate_intervention()

# 关闭数据库连接
conn.close()