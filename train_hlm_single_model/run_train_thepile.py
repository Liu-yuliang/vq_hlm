import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from transformers import get_linear_schedule_with_warmup
from concurrent.futures import ThreadPoolExecutor
import subprocess
from tqdm import tqdm



class StreamingJSONLDataset(Dataset):
    def __init__(self, data_dir, tokenizer, block_size=512):
        """
        初始化方法，读取数据目录、tokenizer 和其他设置
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 获取目录下所有的 JSONL 文件
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
        
        # 记录每个文件的总行数
        self.file_line_counts = {}

        def count_lines(file_path):
            result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
            line_count = int(result.stdout.split()[0])
            return line_count

        # 并行计算每个文件的行数
        with ThreadPoolExecutor() as executor:
            file_paths = [os.path.join(self.data_dir, file) for file in self.files]
            line_counts = executor.map(count_lines, file_paths)

        # 将结果保存到字典
        for file, line_count in zip(self.files, line_counts):
            self.file_line_counts[file] = line_count
        # import pdb; pdb.set_trace()
        self.total_lines = 0
        for value in self.file_line_counts.values():
            self.total_lines += value

        # 初始化文件指针，文件句柄等
        self.files_pointers = {file: 0 for file in self.files}  # 每个文件的当前行索引
        self.current_file_index = 0  # 当前正在处理的文件索引
        self.current_file_handle = None  # 当前打开的文件句柄
        self.current_file_lines = []  # 当前文件所有行的缓存

    def __len__(self):
        """
        返回所有文件的总行数，即数据集的大小
        """
        return self.total_lines

    def __getitem__(self, idx):
        """
        按照索引获取数据，确保每次只打开一个文件
        """
        # 计算应该从哪个文件读取数据
        current_file = None
        current_file_idx = idx
        
        # 找到对应的文件
        for file in self.files:
            if current_file_idx < self.file_line_counts[file]:
                current_file = file
                line_idx = current_file_idx
                break
            current_file_idx -= self.file_line_counts[file]

        # 如果当前文件还没有打开，打开当前文件
        if current_file != self.current_file_index:
            if self.current_file_handle is not None:
                # 关闭上一个文件
                self.current_file_handle.close()

            # 打开当前文件
            file_path = os.path.join(self.data_dir, current_file)
            self.current_file_handle = open(file_path, 'r', encoding='utf-8')
            self.current_file_lines = self.current_file_handle.readlines()
            self.files_pointers[current_file] = 0
            self.current_file_index = current_file

        # 获取当前行的数据
        line_data = self.current_file_lines[line_idx]
        data = json.loads(line_data)

        # 返回 tokenized 的 input_ids，直接作为 tensor
        return torch.tensor(data['input_ids'], dtype=torch.long)

    def __del__(self):
        """
        在数据集对象销毁时，关闭所有打开的文件句柄
        """
        if self.current_file_handle is not None:
            self.current_file_handle.close()

# 初始化模型和tokenizer
model_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/gpt2_wiki"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

vae_model = {'vae_config_path': '../conf/models/residualvq_512.yaml', 'vae_pretrained_model_path': '../runs/residualvq_ckpt1e3_codebookdim512_layer6_epoch3_mp/latest_checkpoint.pt', 'chunk_size': 4} 

training_type = 'after_input_layer'
input_layers = 6
ctx_layers = 9
model = GPT2LMHeadModel.from_pretrained(model_path, input_layers=input_layers, ctx_layers=ctx_layers, vae_model=vae_model, training_type=training_type)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 初始化数据集和数据加载器
data_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/datasets/the_pile_deduplicate/jsonl_1024_train/"
dataset = StreamingJSONLDataset(data_dir, tokenizer)
dataloader = DataLoader(dataset, batch_size=14*4, shuffle=False, num_workers=4)
accumulation_steps = 64

# 优化器和学习率调度器
num_epochs = 1
total_steps = len(dataloader) * num_epochs



if training_type == 'ours':

    embedding_weight = load_saved_params('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/init_modules/VQVAE_embedding_MLP_codebook1024_kaiming_uniform_.pth')['embedding.weight']
    model.transformer.vqvae.embedding.weight = torch.nn.Parameter(embedding_weight)
    model.transformer.vqvae.embedding.weight.requires_grad = False   

    for name, param in model.named_parameters():
        param.requires_grad = False   
    for i in range(input_layers, len(model.transformer.h)):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True  

elif training_type == 'codebook':
    for name, param in model.named_parameters():
        param.requires_grad = False   
    for param in model.transformer.vqvae.parameters():
        param.requires_grad = True  

elif training_type == 'after_input_layer':
    for name, param in model.named_parameters():
        param.requires_grad = False     
    for i in range(input_layers, len(model.transformer.h)):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True  

elif training_type == 'full':
    for name, param in model.named_parameters():
        param.requires_grad = True

elif training_type == 'except_codebook':
    for name, param in model.named_parameters():
        param.requires_grad = True
    for param in model.transformer.vqvae.parameters():
        param.requires_grad = False  

elif training_type == 'only_ctx_layer':
    for name, param in model.named_parameters():
        param.requires_grad = False     
    for i in range(input_layers, ctx_layers):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True  

elif training_type == 'only_output_layer':
    for name, param in model.named_parameters():
        param.requires_grad = False     
    for i in range(ctx_layers, len(model.transformer.h)):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = True  

for name, param in model.named_parameters():
    print(name, param.requires_grad)

# 使用 DataParallel 来利用多GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)


model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

for name, param in model.module.named_parameters():
    print(name, param.requires_grad)

# 训练模型
model.train()
for epoch in range(num_epochs):
    # import pdb; pdb.set_trace()
    total_loss = 0
    for batch_idx, (input_ids) in enumerate(tqdm(dataloader, desc="Training", total=len(dataloader))):
        input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)

        # 训练步骤
        optimizer.zero_grad()

        # GPT2 不需要 labels 显式传递，默认使用 input_ids 作为目标
        back_loss, qualitized_loss, ctx_loss, gpt_loss = model(input_ids, labels=input_ids)
        # import pdb; pdb.set_trace()
        loss = back_loss.loss.mean()
        total_loss += loss

        # 反向传播和优化
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
        # print(123)
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}, Qua Loss: {qualitized_loss.mean()}, Ctx Loss: {ctx_loss.mean()}, GPT Loss: {gpt_loss.mean()}")
            with open('./train_log.jsonl', 'a') as fw:
                # import pdb; pdb.set_trace()
                fw.write(json.dumps({'Epoch': epoch + 1/ num_epochs, 'Batch': batch_idx/len(dataloader), 'Loss': loss.item(), 'Qua Loss': qualitized_loss.mean().item(), 'Ctx Loss': ctx_loss.mean().item(), 'GPT Loss': gpt_loss.mean().item(), 'total step': len(dataloader)}))
                fw.write('\n')
                
        if batch_idx % int(len(dataloader) * 0.003) == 0:
            # 假设 model 是一个 DataParallel 对象
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_save.save_pretrained(f"./the_pile_lr3e4_gpt2_epoch_{epoch + 1}_{batch_idx}")
            tokenizer.save_pretrained(f"./the_pile_lr3e4_gpt2_epoch_{epoch + 1}_{batch_idx}")

    # 打印每个 epoch 的平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

print("Training complete.")
