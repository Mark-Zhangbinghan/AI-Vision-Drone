from ultralytics import YOLO
import torch
import torch.distributed as dist
import os

if __name__ == '__main__':
    # 1. 强行在脚本内部初始化双卡分布式环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        if local_rank == 0:
            print("🚀 双卡 4090 DDP 分布式环境强行初始化成功！")

    # 2. 智能断点续传与初始化逻辑
    last_ckpt_path = '/root/ultralytics-YOLO26/runs/detect/AIVISION_Drone_Fusion/weights/last.pt'
    
    if os.path.exists(last_ckpt_path):
        if os.environ.get('LOCAL_RANK', '0') == '0':
            print(f"🔄 检测到融合训练的历史存档，正在从断点恢复: {last_ckpt_path}")
        model = YOLO(last_ckpt_path)
        model.train(
            resume=True,
            batch=32,         
            workers=8,        
            save_period=1,
            cache='ram'
        )
    else:
        if os.environ.get('LOCAL_RANK', '0') == '0':
            print("🆕 未检测到历史存档，正在加载 YOLOv26s 预训练权重并注入冻结保护...")
        
        model = YOLO('yolo26s.pt') 

        # 🛠️ 【核心修复】用底层原生代码精准冻结前 150 层，完美避开 end2end 冲突
        for k, v in model.model.named_parameters():
            if any(f'.{i}.' in k for i in range(150)):
                v.requires_grad = False

        # 3. 🔥 双卡 4090 极致微调特调（知识融合·安全通道版）
        model.train(
            data='/root/ultralytics-YOLO26/datasets/unified_dataset/data.yaml', 
            epochs=50,          
            batch=32,           
            imgsz=640,          
            workers=8,          
            cache='ram',        
            amp=True,           
            pretrained=True,    
            
            # --- 🛠️ 注意：这里不再写 freeze=150，由上面手工处理 ---
            lr0=0.001,          
            
            # --- 精度专项强化参数 ---
            mosaic=0.5,         
            mixup=0.15,         
            cls=2.0,            
            box=8.5,            
            
            # --- 💾 防御与保存机制 ---
            name='AIVISION_Drone_Fusion', 
            save=True,          
            save_period=1,      
            exist_ok=True,      
            cos_lr=True,        
            optimizer='AdamW'   
        )
        
    # 4. 训练结束后关闭分布式进程
    if dist.is_initialized():
        dist.destroy_process_group()