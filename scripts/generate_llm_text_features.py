import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. 正在加载 Qwen2-VL-7B-Instruct (生成专家级超声文本先验)...")
    # 使用 bfloat16 精度，完美适配 RTX 4090 的算力与显存
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # 🌟 针对当前框架的 6 个域设定特定的病灶目标
    domains = {
        "thyroid": "thyroid nodule",
        "TN3K": "thyroid nodule",
        "BUSI_WHU": "breast mass",
        "BUS-BRA": "breast mass",
        "OTU": "ovarian tumor",       
        "prostate": "prostate gland"  
    }

    prompt_template = (
        "You are an expert radiologist. Describe the general ultrasound imaging "
        "characteristics of a {}. Focus on echogenicity, margin regularity, "
        "and internal structure. Provide the description in 3 concise, professional English sentences."
    )

    domain_texts = {}
    for domain, organ in domains.items():
        print(f"\n正在让 LLM 生成 [{domain}] 的超声特征描述...")
        messages = [{"role": "user", "content": prompt_template.format(organ)}]
        text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = qwen_processor(text=[text_prompt], return_tensors="pt").to(device)
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
        
        # 提取生成的回复文本
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        domain_texts[domain] = output_text.strip()
        print(f"生成的文本: {domain_texts[domain]}")

    print("\n2. 正在加载 BioBERT-large (将文本编码为 1024 维安全格式特征)...")
    # 释放 Qwen 占用的显存，防止两张大模型同时挤爆显存
    del qwen_model
    del qwen_processor
    torch.cuda.empty_cache()

    # 加载原生输出 1024 维的顶级医学文本编码器，并强制只读安全的 safetensors 格式
    bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    bert_model = AutoModel.from_pretrained(
        "dmis-lab/biobert-large-cased-v1.1",
        use_safetensors=True  
    ).to(device)
    bert_model.eval()

    output_dir = "./text_features_llm"
    os.makedirs(output_dir, exist_ok=True)

    print("\n3. 预先计算辅助属性特征 (Appearance, Boundary, Exclusion, Task)...")
    # 定义 4 个额外的属性文本，用于与 LLM 的 target 特征进行特征空间融合
    attributes_text = [
        "ultrasound lesion appearance and internal texture",  # appearance
        "lesion boundary, margin and contour",                # boundary
        "normal background tissue exclusion and speckle noise", # exclusion
        "segmentation task of the target lesion"              # task
    ]
    attr_inputs = bert_tokenizer(attributes_text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        attr_outputs = bert_model(**attr_inputs)
        # 提取这四个属性的 [CLS] 作为语义方向基准向量 [4, 1024]
        attr_feats = attr_outputs.last_hidden_state[:, 0, :].cpu()

    print("\n4. 正在编码并保存融合后的密集语义特征 .pt 文件...")
    GROUP_SUMMARY_COUNT = 5  

    for domain, text in domain_texts.items():
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            
            # 获取所有 token 的特征: [1, Seq_len, 1024]
            last_hidden_state = outputs.last_hidden_state[0].cpu() 
            attention_mask = inputs["attention_mask"][0].cpu().bool()
            
            # 过滤掉 padding，仅保留有效 token 的特征
            valid_feats = last_hidden_state[attention_mask] 
            
            # 提取 [CLS] 作为全局特征 (即你的 Target)
            global_feat = valid_feats[0].unsqueeze(0)       # Shape: [1, 1024]
            # 提取句子本身的密集细粒度特征
            fine_feats = valid_feats[1:-1]                  
            
        # ==========================================
        # 核心修改：基于权重的属性语义融合 (Semantic Blending)
        # ==========================================
        # Token 0: 100% Target 
        token_0 = global_feat
        
        # Token 1: Target * 0.8 + Appearance * 0.2
        token_1 = 0.8 * global_feat + 0.2 * attr_feats[0].unsqueeze(0)
        
        # Token 2: Target * 0.8 + Boundary * 0.2
        token_2 = 0.8 * global_feat + 0.2 * attr_feats[1].unsqueeze(0)
        
        # Token 3: Target * 0.8 + Exclusion * 0.2
        token_3 = 0.8 * global_feat + 0.2 * attr_feats[2].unsqueeze(0)
        
        # Token 4: Target * 0.8 + Task * 0.2
        token_4 = 0.8 * global_feat + 0.2 * attr_feats[3].unsqueeze(0)
        
        # 拼接成拥有 5 种不同侧重语义的 Group Summary Tokens [5, 1024]
        group_summary_tokens = torch.cat([token_0, token_1, token_2, token_3, token_4], dim=0) 
        
        # 拼接成最终模型期望的格式: 前 5 个是融合后的 summary, 后面跟着密集子词特征
        final_text_features = torch.cat([group_summary_tokens, fine_feats], dim=0) # [5 + 细粒度Token数, 1024]
        total_tokens = final_text_features.shape[0]
        
        # 构造向下兼容的伪造掩码与索引，供 Attention 层使用
        text_mask = torch.ones(total_tokens, dtype=torch.long)
        token_is_group_summary = torch.cat([
            torch.ones(GROUP_SUMMARY_COUNT, dtype=torch.long),
            torch.zeros(fine_feats.shape[0], dtype=torch.long)
        ], dim=0)
        
        save_dict = {
            "text_features": final_text_features,                
            "text_mask": text_mask,                              
            "text_group_summary_count": GROUP_SUMMARY_COUNT,     
            "token_is_group_summary": token_is_group_summary,    
            "raw_text": text,                                    
            "hidden_dim": 1024                                   
        }
            
        save_path = os.path.join(output_dir, f"text_features_{domain}.pt")
        torch.save(save_dict, save_path)
        print(f"已保存 [{domain}] 特征 -> {save_path} | 特征矩阵 Shape: {final_text_features.shape}")

    print("\n✅ 融合式 (Target+Attributes) 密集文本特征生成完毕！")

if __name__ == "__main__":
    main()