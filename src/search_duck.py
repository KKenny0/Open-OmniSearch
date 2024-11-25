import json
import os
import time
import requests
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS
from loguru import logger

retry_attempt = 5
max_results = 5


def search_text_by_text(text):
    """使用 DuckDuckGo 搜索内容

    Args:
        text (str): 搜索文本

    Returns:
        list[dict[str, str]]: list of dictionaries storing the body, href and title
    """
    with DDGS() as ddgs:

        for i in range(retry_attempt):
            try:
                ddgs_gen = ddgs.text(text,
                                     safesearch='Off',
                                     max_results=max_results)
                return ddgs_gen

            except Exception as e:
                print(f"Attempt {i + 1} failed: {e}")
                if i < retry_attempt - 1:
                    time.sleep(2)  # 等待 2 秒后重试
                else:
                    logger.error("All retries failed. DuckDuckGo search error.")
                    raise ValueError(e)


def search_image_by_text(text):
    """使用 DuckDuckGo 搜索图片

    Args:
        text (str): 搜索文本

    Returns:
        list[dict[str, str]]: list of dictionaries storing
            the height, image, source, thumbnail, title, url and width
    """
    with DDGS() as ddgs:
        for i in range(retry_attempt):
            try:
                ddgs_gen = ddgs.images(text,
                                       safesearch='Off',
                                       max_results=max_results)
                print(f"{ddgs_gen=}")
                return ddgs_gen[0]

            except Exception as e:
                print(f"Attempt {i + 1} failed: {e}")
                if i < retry_attempt - 1:
                    time.sleep(2)  # 等待 2 秒后重试
                else:
                    logger.error("All retries failed. DuckDuckGo search error.")
                    raise ValueError(e)


def parse_image_search_result_by_text(search_result, save_path, idx, conversation_num):
    """
    解析 DuckDuckGo 搜索索图片结果

    Args:
        search_result (generator): DuckDuckGo 搜索图片结果
        save_path (str): 图片保存路径
        idx (int): 图片序号
        conversation_num (int): 对话序号

    Returns:
        tuple[list[str, str], list[str]]: (图片 URL 列表, 图像存储地址), 图像标题信息
    """
    search_images = []
    search_texts = []

    # 获取原始图片 URL
    image_url = search_result['image']

    # 下载并保存图片
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # 验证 HTTP 状态码
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)

        save_image_path = os.path.join(save_path,
                                       '{}_{}_{}.png'.format(idx,
                                                             conversation_num,
                                                             search_result.get("position", "0")))
        image.save(save_image_path, format='PNG')

        # 添加图像路径和描述到列表中
        search_images.append((image_url, save_image_path))
        search_texts.append(search_result.get('title', ''))  # 获取图像的标题和来源

    except Exception as e:
        logger.error(f"Failed to download or save image from {image_url}: {e}")

    return search_images, search_texts


def fine_search(query, search_type, save_path, dataset_name, idx, conversation_num):
    """
    基于文本的深度搜索

    Args:
        query (str): 搜索文本
        search_type (str): 搜索类型
        save_path (str): 图片保存路径
        dataset_name (str): 图片数据集名称
        idx (int): 图片序号
        conversation_num (int): 对话序号

    Returns:
        tuple[list[tuple[str, str]], list[str]]: (图片 URL 列表, 图像存储路径), 图片标题信息
    """
    if search_type == 'text_search_text':
        search_results = search_text_by_text(query)
        search_texts = []

        for item in search_results:
            text_data = ''
            if 'title' in item:
                text_data += item['title']
            if 'body' in item:
                text_data += item['body']
            search_texts.append(text_data)

        logger.debug(f"text_search_text: {search_texts=}")
        return [], search_texts

    elif search_type == 'img_search_img':
        image_search_path = os.path.join(save_path, dataset_name, 'image_search_res_{}.json'.format(idx))
        if os.path.exists(image_search_path):
            print("Image Done!!!")
            with open(image_search_path, 'r') as f_tmp:
                search_result = json.load(f_tmp)
                search_images, search_texts = parse_image_search_result_by_text(search_result,
                                                                                save_path,
                                                                                idx,
                                                                                conversation_num)
                if len(search_texts) == 0:
                    print('Extra image search!')
                    search_result = search_image_by_text(query)
                    search_images, search_texts = parse_image_search_result_by_text(search_result,
                                                                                    save_path,
                                                                                    idx,
                                                                                    conversation_num)

                logger.debug(f"img_search_img: {search_images=}\n{search_texts=}")
                return search_images, search_texts
        else:
            search_result = search_image_by_text(query)
            search_images, search_texts = parse_image_search_result_by_text(search_result,
                                                                            save_path,
                                                                            idx,
                                                                            conversation_num)
            logger.debug(f"text_search_img: {search_images=}\n{search_texts=}")
            return search_images, search_texts

    else:
        search_results = search_image_by_text(query)
        search_images, search_texts = parse_image_search_result_by_text(search_results,
                                                                        save_path,
                                                                        idx,
                                                                        conversation_num)

        return search_images, search_texts
