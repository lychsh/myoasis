import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# 连接数据库
conn = sqlite3.connect('./../result/test.db')

# 获取所有唯一的时间步
time_steps_query = """
SELECT DISTINCT created_at AS time_step
FROM trace
ORDER BY time_step
"""
time_steps_df = pd.read_sql_query(time_steps_query, conn)
time_steps = max(time_steps_df['time_step'].tolist())
print("Time Steps:", time_steps)

# post_id 列表
post_ids = [1, 2, 3] 

# 存储所有 post 的序列数据
all_scale_data = []
all_depth_data = []

for post_id in post_ids:
    scale_data = []
    depth_data = []
    
    for ts in range(time_steps):
        # 传播规模：累积到该时间步的唯一用户数
        scale_query = f"""
        SELECT COUNT(DISTINCT u.user_id) AS scale
        FROM (
            SELECT l.user_id FROM like l WHERE l.post_id = {post_id} AND l.created_at <= {ts}
            UNION
            SELECT d.user_id FROM dislike d WHERE d.post_id = {post_id} AND d.created_at <= {ts}
            UNION
            SELECT c.user_id FROM comment c WHERE c.post_id = {post_id} AND c.created_at <= {ts}
            UNION
            SELECT cl.user_id FROM comment_like cl
            JOIN comment c ON cl.comment_id = c.comment_id
            WHERE c.post_id = {post_id} AND cl.created_at <= {ts} AND c.created_at <= {ts}
            UNION
            SELECT cd.user_id FROM comment_dislike cd
            JOIN comment c ON cd.comment_id = c.comment_id
            WHERE c.post_id = {post_id} AND cd.created_at <= {ts} AND c.created_at <= {ts}
            UNION
            SELECT p.user_id FROM post p WHERE p.original_post_id = {post_id} AND p.created_at <= {ts}
            UNION
            SELECT r.user_id FROM report r WHERE r.post_id = {post_id} AND r.created_at <= {ts}
            UNION
            SELECT tr.user_id FROM trace tr
            WHERE tr.action = 'refresh' AND tr.created_at <= {ts}
            AND json_extract(tr.info, '$.posts[0].post_id') = {post_id}
        ) u
        """
        scale_df = pd.read_sql_query(scale_query, conn)
        scale = scale_df['scale'].iloc[0] if not scale_df.empty else 0
        scale_data.append(scale)
        
        # 传播深度：累积到该时间步的最大层级
        depth_query = f"""
        WITH RECURSIVE post_chain (post_id, depth) AS (
            SELECT {post_id}, 0
            WHERE EXISTS (SELECT 1 FROM post WHERE post_id = {post_id} AND created_at <= {ts})
            UNION ALL
            SELECT p.post_id, pc.depth + 1
            FROM post p JOIN post_chain pc ON p.original_post_id = pc.post_id
            WHERE p.created_at <= {ts}
        ),
        interactions (depth) AS (
            SELECT 1 FROM comment WHERE post_id = {post_id} AND created_at <= {ts}
            UNION ALL
            SELECT 1 FROM post WHERE original_post_id = {post_id} AND created_at <= {ts}
            UNION ALL
            SELECT 1 FROM like WHERE post_id = {post_id} AND created_at <= {ts}
            UNION ALL
            SELECT 1 FROM dislike WHERE post_id = {post_id} AND created_at <= {ts}
            UNION ALL
            SELECT 1 FROM report WHERE post_id = {post_id} AND created_at <= {ts}
            UNION ALL
            SELECT 2 FROM comment_like cl JOIN comment c ON cl.comment_id = c.comment_id
            WHERE c.post_id = {post_id} AND cl.created_at <= {ts} AND c.created_at <= {ts}
            UNION ALL
            SELECT 2 FROM comment_dislike cd JOIN comment c ON cd.comment_id = c.comment_id
            WHERE c.post_id = {post_id} AND cd.created_at <= {ts} AND c.created_at <= {ts}
        )
        SELECT COALESCE(MAX(depth), 0) AS max_depth
        FROM (
            SELECT depth FROM post_chain WHERE post_id != {post_id}
            UNION ALL
            SELECT depth FROM interactions
        )
        """
        depth_df = pd.read_sql_query(depth_query, conn)
        depth = depth_df['max_depth'].iloc[0] if not depth_df.empty else 0
        depth_data.append(depth)
    
    all_scale_data.append(scale_data)
    all_depth_data.append(depth_data)

# 计算平均值
avg_scale_data = pd.DataFrame(all_scale_data).mean(axis=0).tolist()
avg_depth_data = pd.DataFrame(all_depth_data).mean(axis=0).tolist()

# 创建 DataFrame 保存平均序列
df = pd.DataFrame({
    'time_step': time_steps,
    'avg_spread_scale': avg_scale_data,
    'avg_spread_depth': avg_depth_data
})

# 输出序列数据
print("Average Spread Scale and Depth Time Series:")
print(df)

# 保存到 CSV
df.to_csv('avg_spread_sequences.csv', index=False)
print("\nAverage sequences saved to 'avg_spread_sequences.csv'")

# 可视化：传播规模图
plt.figure(figsize=(10, 6))
plt.plot(df['time_step'], df['avg_spread_scale'], color='tab:red', marker='o', label='Average Spread Scale')
plt.xlabel('Time Step')
plt.ylabel('Average Spread Scale')
plt.title('Time Series of Average Spread Scale for Multiple Posts')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('avg_spread_scale.png', dpi=300, bbox_inches='tight')
print("Spread scale plot saved to 'avg_spread_scale.png'")
plt.show()

# 可视化：传播深度图
plt.figure(figsize=(10, 6))
plt.plot(df['time_step'], df['avg_spread_depth'], color='tab:blue', marker='s', label='Average Spread Depth')
plt.xlabel('Time Step')
plt.ylabel('Average Spread Depth')
plt.title('Time Series of Average Spread Depth for Multiple Posts')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('avg_spread_depth.png', dpi=300, bbox_inches='tight')
print("Spread depth plot saved to 'avg_spread_depth.png'")
plt.show()

# 关闭数据库连接
conn.close()