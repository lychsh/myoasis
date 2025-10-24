import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import re
from collections import Counter

# 设置中文字体 
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("中文字体设置失败，使用默认字体")

# 数据准备
with open('./../data/fake_info/fake_info.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 中文文本预处理函数
def preprocess_chinese_text(text):
    """中文文本预处理"""
    if not isinstance(text, str):
        return ""
    
    # 去除特殊字符和标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    # 使用jieba分词
    words = jieba.cut(text)
    
    # 过滤停用词和短词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这个', '那个','一直','以及', '我们'}
    filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    return ' '.join(filtered_words)

# 准备数据
real_texts = []
fake_texts = []
ai_fake_texts = []
categories = []

for item in data['data']:
    real_texts.append(preprocess_chinese_text(item['real_info']))
    fake_texts.append(preprocess_chinese_text(item['fake_info']))
    ai_fake_texts.append(preprocess_chinese_text(item['ai_fake_info']))
    categories.append(item['category'])

print("数据预处理完成")
print(f"官方内容样本数: {len(real_texts)}")
print(f"虚假信息样本数: {len(fake_texts)}")
print(f"AI虚假信息样本数: {len(ai_fake_texts)}")

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=1,
    max_df=0.8,
    ngram_range=(1, 2)  # 包含单字和双字词组
)

# 合并所有文本进行拟合
all_texts = real_texts + fake_texts + ai_fake_texts
vectorizer.fit(all_texts)

# 转换不同类型的文本
real_tfidf = vectorizer.transform(real_texts)
fake_tfidf = vectorizer.transform(fake_texts)
ai_fake_tfidf = vectorizer.transform(ai_fake_texts)

# 计算平均TF-IDF分数
real_mean_tfidf = np.array(real_tfidf.mean(axis=0)).flatten()
fake_mean_tfidf = np.array(fake_tfidf.mean(axis=0)).flatten()
ai_fake_mean_tfidf = np.array(ai_fake_tfidf.mean(axis=0)).flatten()

# 获取特征词汇
feature_names = vectorizer.get_feature_names_out()

# 计算区分度指标
def calculate_discrimination_scores(real_scores, fake_scores, feature_names):
    """计算词汇在两类文本中的区分度"""
    discrimination_data = []
    for i, word in enumerate(feature_names):
        real_score = real_scores[i]
        fake_score = fake_scores[i]
        
        # 区分度定义为两类文本TF-IDF分数的绝对差异
        discrimination_score = abs(real_score - fake_score)
        
        # 方向性：正数表示在官方内容中更突出，负数表示在虚假信息中更突出
        direction = 1 if real_score > fake_score else -1
        
        discrimination_data.append({
            'word': word,
            'discrimination_score': discrimination_score,
            'direction': direction,
            'real_tfidf': real_score,
            'fake_tfidf': fake_score
        })
    
    return pd.DataFrame(discrimination_data)

# 计算官方内容 vs 普通虚假信息的区分度
discrimination_df = calculate_discrimination_scores(real_mean_tfidf, fake_mean_tfidf, feature_names)

# 计算官方内容 vs AI虚假信息的区分度
discrimination_ai_df = calculate_discrimination_scores(real_mean_tfidf, ai_fake_mean_tfidf, feature_names)

# 选择前10%的高区分度关键词
top_percentage = 0.1
top_n = int(len(discrimination_df) * top_percentage)

# 对普通虚假信息的高区分度词汇
top_discrimination_words = discrimination_df.nlargest(top_n, 'discrimination_score')
top_real_words = top_discrimination_words[top_discrimination_words['direction'] == 1]
top_fake_words = top_discrimination_words[top_discrimination_words['direction'] == -1]

# 对AI虚假信息的高区分度词汇
top_discrimination_ai_words = discrimination_ai_df.nlargest(top_n, 'discrimination_score')
top_real_ai_words = top_discrimination_ai_words[top_discrimination_ai_words['direction'] == 1]
top_fake_ai_words = top_discrimination_ai_words[top_discrimination_ai_words['direction'] == -1]

print(f"\n=== 官方内容 vs 普通虚假信息 ===")
print(f"高区分度词汇总数: {len(top_discrimination_words)}")
print(f"官方内容特征词汇数: {len(top_real_words)}")
print(f"虚假信息特征词汇数: {len(top_fake_words)}")

print(f"\n=== 官方内容 vs AI虚假信息 ===")
print(f"高区分度词汇总数: {len(top_discrimination_ai_words)}")
print(f"官方内容特征词汇数: {len(top_real_ai_words)}")
print(f"AI虚假信息特征词汇数: {len(top_fake_ai_words)}")



# 1. 官方内容 vs 普通虚假信息 - 高区分度词汇
plt.figure(figsize=(12, 8))
top_combined = pd.concat([
    top_real_words.head(10).assign(type='Official'),
    top_fake_words.head(10).assign(type='Fake')
])
sns.barplot(data=top_combined, x='discrimination_score', y='word', hue='type')
plt.title('官方 vs 虚假信息top 10%区分度关键词', fontsize=14, fontweight='bold')
plt.xlabel('区分度分数')
plt.ylabel('关键字')
plt.tight_layout()
plt.savefig('./imgs/official_fake_discriminative_keywords.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 官方内容 vs AI虚假信息 - 高区分度词汇
plt.figure(figsize=(12, 8))
top_combined_ai = pd.concat([
    top_real_ai_words.head(10).assign(type='Official'),
    top_fake_ai_words.head(10).assign(type='AI Fake')
])
sns.barplot(data=top_combined_ai, x='discrimination_score', y='word', hue='type')
plt.title('官方 vs AI虚假信息top 10%区分度关键词', fontsize=14, fontweight='bold')
plt.xlabel('区分度分数')
plt.ylabel('关键字')
plt.tight_layout()
plt.savefig('./imgs/official_aifake_ai_discriminative_keywords.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 不同类型文本的特征词汇分布
plt.figure(figsize=(10, 6))
type_comparison = pd.DataFrame({
    'Type': ['Official', 'Fake', 'AI Fake'],
    'Feature Count': [
        len(top_real_words),
        len(top_fake_words), 
        len(top_fake_ai_words)
    ]
})
sns.barplot(data=type_comparison, x='Type', y='Feature Count', hue='Type', legend=False)
plt.title('高区分特征统计', fontsize=14, fontweight='bold')
plt.xlabel('类别')
plt.ylabel('特征词汇数量')
plt.tight_layout()
plt.savefig('./imgs/feature_count.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. TF-IDF分数对比
plt.figure(figsize=(12, 8))
sample_words = list(top_real_words.head(5)['word']) + list(top_fake_words.head(5)['word'])
comparison_data = []
for word in sample_words:
    if word in discrimination_df['word'].values:
        row = discrimination_df[discrimination_df['word'] == word].iloc[0]
        comparison_data.append({'word': word, 'score': row['real_tfidf'], 'type': 'Official'})
        comparison_data.append({'word': word, 'score': row['fake_tfidf'], 'type': 'Fake'})
comparison_df = pd.DataFrame(comparison_data)
sns.barplot(data=comparison_df, x='score', y='word', hue='type')
plt.title('TF-IDF分数对比', fontsize=14, fontweight='bold')
plt.xlabel('TF-IDF分数')
plt.ylabel('关键字')
plt.tight_layout()
plt.savefig('./imgs/tfidf_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细的关键词分析结果
print("\n" + "="*80)
print("Detailed Keyword Analysis Report")
print("="*80)

print(f"\n1. Official Content Keywords (Top 10):")
print("-" * 50)
for i, (_, row) in enumerate(top_real_words.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['word']:12} | Discrimination: {row['discrimination_score']:.4f} | "
          f"Official TF-IDF: {row['real_tfidf']:.4f} | Fake TF-IDF: {row['fake_tfidf']:.4f}")

print(f"\n2. Fake News Keywords (Top 10):")
print("-" * 50)
for i, (_, row) in enumerate(top_fake_words.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['word']:12} | Discrimination: {row['discrimination_score']:.4f} | "
          f"Official TF-IDF: {row['real_tfidf']:.4f} | Fake TF-IDF: {row['fake_tfidf']:.4f}")

print(f"\n3. AI Fake News Keywords (Top 10):")
print("-" * 50)
for i, (_, row) in enumerate(top_fake_ai_words.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['word']:12} | Discrimination: {row['discrimination_score']:.4f} | "
          f"Official TF-IDF: {row['real_tfidf']:.4f} | AI Fake TF-IDF: {row['fake_tfidf']:.4f}")

# 构建特征词典
feature_dict = {
    'official_keywords': list(top_real_words['word']),
    'fake_keywords': list(top_fake_words['word']),
    'ai_fake_keywords': list(top_fake_ai_words['word'])
}

print(f"\n4. Feature Dictionary Statistics:")
print("-" * 50)
print(f"Official Content Keywords: {len(feature_dict['official_keywords'])}")
print(f"Fake News Keywords: {len(feature_dict['fake_keywords'])}")
print(f"AI Fake News Keywords: {len(feature_dict['ai_fake_keywords'])}")

# 语义分析总结
print(f"\n5. Semantic Analysis Summary:")
print("-" * 50)
print("Official Content Features: Policy terms, professional expressions, systematic descriptions")
print("   Example words:", ", ".join(top_real_words.head(5)['word'].tolist()))
print("\nFake News Features: Emotional vocabulary, conspiracy expressions, accusatory language")
print("   Example words:", ", ".join(top_fake_words.head(5)['word'].tolist()))
print("\nAI Fake News Features: Technical descriptions, systematic conspiracies, data-related vocabulary")
print("   Example words:", ", ".join(top_fake_ai_words.head(5)['word'].tolist()))

# 保存结果到文件
results = {
    'analysis_summary': {
        'total_features': len(feature_names),
        'official_vs_fake': {
            'total_discriminative': len(top_discrimination_words),
            'official_keywords': len(top_real_words),
            'fake_keywords': len(top_fake_words)
        },
        'official_vs_ai_fake': {
            'total_discriminative': len(top_discrimination_ai_words),
            'official_keywords': len(top_real_ai_words),
            'ai_fake_keywords': len(top_fake_ai_words)
        }
    },
    'feature_dictionary': feature_dict
}

# 保存为JSON文件
import json
with open('tfidf_analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to 'tfidf_analysis_results.json'")
print(f"Visualization saved to 'tfidf_analysis_results.png'")