import sqlite3
import numpy as np
all_posts = []
all_comments = []

contract_content = ''
class Post():
    def __init__(self, id, poster, content, likes, dislikes):
        self.post_id = id
        self.poster = poster
        self.content = content
        self.likes = likes
        self.dislikes = dislikes
    
    def __str__(self):
        return f"Post {self.post_id}: {self.poster} - {self.content} (likes: {self.likes}, dislikes: {self.dislikes})"

class Comments():
    def __init__(self, user_id, poster_id, content):
        self.user_id = user_id
        self.poster_id = poster_id
        self.content = content
    
    def __str__(self):
        return f"Comment by {self.user_id} on post {self.poster_id}: {self.content}"
        

def extract_posts_simple(db_path):
    global contract_content
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 提取帖子内容
        query = """
        SELECT 
            u.user_name,
            u.name,
            p.content,
            p.num_likes,
            p.num_dislikes
        FROM post p 
        JOIN user u ON p.user_id = u.user_id
        """
        
        cursor.execute(query)
        posts = cursor.fetchall()

        for i, (user_name, name, content, likes, dislikes) in enumerate(posts, 1):
            post = Post(i, user_name, content, likes, dislikes)
            all_posts.append(post)
            contract_content += content


        with open('all_posts.txt', 'w') as file:
            file.write(contract_content)
        
        # 提取评论
        comment_query = """
        SELECT 
            u.user_name,
            c.content,
            c.post_id,
            p.content as post_content
        FROM comment c
        JOIN user u ON c.user_id = u.user_id
        JOIN post p ON c.post_id = p.post_id
        ORDER BY c.post_id, c.comment_id
        """
        
        cursor.execute(comment_query)
        comments = cursor.fetchall()
        
        if comments:
            current_post_id = None
            for user_name, content, post_id, post_content in comments:
                # 如果是新的帖子，显示帖子信息
                comment = Comments(user_name, post_id, content)
                all_comments.append(comment)
                # if current_post_id != post_id:
                #     current_post_id = post_id
                #     print(f"\n--- 评论针对帖子 ID {post_id} ---")
                #     print(f"帖子内容: {post_content}")
                #     print("-" * 40)
                # 
                # print(f"{user_name}: {content}")
        conn.close()
        
    except Exception as e:
        print(f"ERR: {e}")


def print_infor():
    print("=" * 60)
    print("所有帖子:")
    print("=" * 60)
    for post in all_posts:
        print(post)
    
    print("\n" + "=" * 60)
    print("所有评论:")
    print("=" * 60)
    for comment in all_comments:
        print(comment)


def pagerank():
    pass


if __name__ == "__main__":
    extract_posts_simple('test.db')
    # print_infor()

