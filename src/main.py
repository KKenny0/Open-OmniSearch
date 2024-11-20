import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json
import asyncio
import argparse
from agent import QAAgent
from conversation_manager import ConversationManager


# 初始化 conversation_manager
model = "gpt-4o"
GPT_API_KEY = ""
headers = {
    "Authorization": f"Bearer {GPT_API_KEY}"
}
meta_save_path = './docs/'  # 保存地址

    
# 定义线程锁
write_lock = threading.Lock()


# 线程安全地写入文件的函数
def safe_write(file_path, data):
    with write_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# 将 process_item 改写为调用 ConversationManager 来处理每个数据项
def process_item(item, conversation_manager, meta_save_path, dataset_name):
    input_question = item['question']
    idx = item['question_id']
    image_url = item['image_url']

    # 调用 conversation_manager 的 manage_conversation 方法
    answer, current_message = conversation_manager.manage_conversation(
        input_question=input_question, image_url=image_url, idx=idx
    )
    
    # 将结果保存到 item 中
    item['prediction'] = answer
    # 保存结果
    output_path = os.path.join(meta_save_path, dataset_name, "output_from_llm.jsonl")
    safe_write(output_path, item)


def main(test_dataset, dataset_name, meta_save_path):
    """
    main 函数，结合 AutoGen 的 ConversationManager 和线程池处理
    """

    qa_agent = QAAgent()

    with open(test_dataset, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]

    # 如果文件存在，过滤已处理的项目
    output_path = os.path.join(meta_save_path, dataset_name, "output_from_llm.jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r") as fin:
            done_id = [json.loads(data)['question_id'] for data in fin.readlines()]
            datas = [data for data in datas if data['question_id'] not in done_id]

    # 设置 save_path 并创建文件夹
    save_path = os.path.join(meta_save_path, dataset_name, "search_images_gpt4v")
    os.makedirs(save_path, exist_ok=True)

    conversation_manager = ConversationManager(qa_agent=qa_agent, dataset_name=dataset_name, save_path=save_path)
    for item in datas:
        process_item(item, conversation_manager, meta_save_path, dataset_name)
    

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行指定的数据集")
    parser.add_argument("--test_dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--meta_save_path", type=str, required=True, help="存储路径")

    args = parser.parse_args()

    # 调用 main 函数并传递解析后的参数
    main(args.test_dataset, args.dataset_name, args.meta_save_path)
