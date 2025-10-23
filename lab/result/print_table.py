import sqlite3

def analyze_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有表
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    
    tables = [table[0] for table in cursor.fetchall()]
    
    print(f"数据库 '{db_path}' 包含以下表:")
    print("=" * 50)
    
    for table in tables:
        print(f"\n表名: {table}")
        print("-" * 30)
        
        # 获取列信息
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        print("表结构:")
        for col in columns:
            pk_indicator = " (主键)" if col[5] else ""
            print(f"  {col[1]} ({col[2]}){pk_indicator}")
        
        # 获取数据样例
        cursor.execute(f"SELECT * FROM {table} LIMIT 3")
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        
        print(f"\n前3行数据 (共{len(rows)}行):")
        print("列名:", col_names)
        for i, row in enumerate(rows, 1):
            print(f"第{i}行: {row}")
        
        # 显示总行数
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]
        print(f"表 '{table}' 总行数: {total_rows}")
    
    conn.close()

# 执行分析
analyze_database('test.db')