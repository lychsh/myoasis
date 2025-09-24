from transformers import AutoTokenizer, AutoModel
import torch
import os

def test_twhin_bert_basic():
    # 模型本地路径
    model_path = "/home/liying/my_models/twhin-bert-base"
    
    try:
        # 加载tokenizer和模型
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        print("Loading model...")
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        
        # 测试文本
        test_text = "Hello, this is a test sentence for TWIN-BERT model."
        
        # 编码文本
        print("Tokenizing text...")
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
        
        # 模型推理
        print("Running model inference...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取embedding
        embeddings = outputs.last_hidden_state
        print(f"Input text: {test_text}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"First 10 dimensions of first token: {embeddings[0, 0, :10]}")
        
        print("✅ Model loaded and tested successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_twhin_bert_basic()