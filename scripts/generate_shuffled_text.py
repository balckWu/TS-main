import os
import sys
import torch
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ============================================================
# 自动定位项目根目录，确保生成的文件一定在 TaskSegmentV3 根目录下
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def shuffle_text(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def main():
    print("=" * 60)
    print("🚀 开始执行：专家文本乱序 (Shuffled Text) 生成脚本")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42)  # 固定随机种子，确保每次乱序结果一致且可复现

    # 设置绝对保存路径
    output_dir = ROOT / "text_features_shuffled"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 目标保存文件夹 (绝对路径): {output_dir.resolve()}")

    # 1. 原始通顺文本
    custom_domain_texts = {
        "thyroid": "A typical malignant thyroid nodule appears as a markedly hypoechoic solid mass with irregular, microlobulated margins. It often exhibits internal microcalcifications and a taller-than-wide shape. The background tissue shows normal thyroid echotexture.",
        "TN3K": "A typical malignant thyroid nodule appears as a markedly hypoechoic solid mass with irregular, microlobulated margins. It often exhibits internal microcalcifications and a taller-than-wide shape. The background tissue shows normal thyroid echotexture.",
        "BUSI_WHU": "Breast masses often present as hypoechoic lesions with irregular boundaries and posterior acoustic shadowing. The internal echotexture is heterogeneous, indicating complex tissue composition.",
        "BUS-BRA": "Breast masses often present as hypoechoic lesions with irregular boundaries and posterior acoustic shadowing. The internal echotexture is heterogeneous, indicating complex tissue composition.",
        "OTU": "Ovarian tumors in ultrasound can present as complex cystic or solid masses with irregular septations and papillary projections. The echogenicity varies depending on the fluid or tissue content.",
        "prostate": "Prostate gland lesions typically manifest as focal hypoechoic areas within the peripheral zone. The margins can be poorly defined, contrasting with the normal homogeneous background of the prostate capsule."
    }

    # 2. 将文本全部转化为乱序
    shuffled_texts = {domain: shuffle_text(text) for domain, text in custom_domain_texts.items()}
    
    print("\n🔍 文本乱序效果检查：")
    for domain in shuffled_texts:
        print(f" [{domain} 原文]: {custom_domain_texts[domain][:60]}...")
        print(f" [{domain} 乱序]: {shuffled_texts[domain][:60]}...\n")

    # 3. 加载 BioBERT
    print("⏳ 正在加载 BioBERT-large (由于是从本地缓存读取，可能需要几秒钟)...")
    bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    bert_model = AutoModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1", use_safetensors=True).to(device)
    bert_model.eval()

    # 4. 预计算属性特征 (保留正常语义，只破坏 Target)
    print("⏳ 正在预计算 4 个辅助属性的特征...")
    attributes_text = [
        "ultrasound lesion appearance and internal texture",  
        "lesion boundary, margin and contour",                
        "normal background tissue exclusion and speckle noise", 
        "segmentation task of the target lesion"              
    ]
    attr_inputs = bert_tokenizer(attributes_text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        attr_outputs = bert_model(**attr_inputs)
        attr_feats = attr_outputs.last_hidden_state[:, 0, :].cpu()

    GROUP_SUMMARY_COUNT = 5  

    # 5. 针对乱序文本进行编码并保存
    print("\n💾 正在保存乱序特征文件 (.pt)...")
    for domain, text in shuffled_texts.items():
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state[0].cpu() 
            attention_mask = inputs["attention_mask"][0].cpu().bool()
            valid_feats = last_hidden_state[attention_mask] 
            
            global_feat = valid_feats[0].unsqueeze(0)       
            fine_feats = valid_feats[1:-1]                  
            
        token_0 = global_feat
        token_1 = 0.8 * global_feat + 0.2 * attr_feats[0].unsqueeze(0)
        token_2 = 0.8 * global_feat + 0.2 * attr_feats[1].unsqueeze(0)
        token_3 = 0.8 * global_feat + 0.2 * attr_feats[2].unsqueeze(0)
        token_4 = 0.8 * global_feat + 0.2 * attr_feats[3].unsqueeze(0)
        
        group_summary_tokens = torch.cat([token_0, token_1, token_2, token_3, token_4], dim=0) 
        final_text_features = torch.cat([group_summary_tokens, fine_feats], dim=0) 
        
        total_tokens = final_text_features.shape[0]
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
            "raw_text": text,  # 这里的 text 已经是乱序的了                             
            "hidden_dim": 1024                                   
        }
            
        save_path = output_dir / f"text_features_{domain}.pt"
        torch.save(save_dict, str(save_path))
        print(f"  ✅ 已生成: {save_path.name}")

    print("\n🎉 成功！所有乱序文本特征均已生成在 text_features_shuffled 文件夹下。")

if __name__ == "__main__":
    main()