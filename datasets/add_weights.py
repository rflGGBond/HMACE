import random
import os
from tqdm import tqdm

if __name__ == "__main__":
    # 要处理的文件名列表
    graphs = ["BA3000/BA3000.txt", "ER3000/ER3000.txt", "RG3000/RG3000.txt", "WS3000/WS3000.txt", 
                "email-Eu-core/email-Eu-core.txt", "Email-EuAll/Email-EuAll.txt", "HR/HR.txt",
                "p2p-Gnutella31/p2p-Gnutella31.txt", "soc-Epinions1/soc-Epinions1.txt"]
    
    # 权重列表
    weights = [0.01, 0.05, 0.2]

    # 遍历每个文件
    for graph in graphs:
        if not os.path.exists(graph):
            print(f"⚠️ 跳过：未找到文件 {graph}")
            continue
    
        # 读取所有边
        with open(graph, 'r') as f:
            lines = f.readlines()
        
        print(f"\n正在处理 {graph} ({len(lines)} 条边)...")

        with open(graph, 'w') as f:
            for line in tqdm(lines, desc=f"处理 {graph}", ncols=80):
                parts = line.strip().split()
                if len(parts) == 2:
                    u, v = parts
                    w = random.choice(weights)
                    f.write(f"{u} {v} {w}\n")
                
        print(f"✅ 已为 {graph} 添加随机权重")
    
    print("\n🎯 全部文件处理完成！")