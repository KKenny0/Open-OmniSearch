import re
import requests
from io import BytesIO
from typing import Tuple, List
from loguru import logger

from .prompt import *
from .search import SearchService

# Initialize the search service as a module-level singleton
_search_service = None


def get_search_service():
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


class ConversationManager:
    def __init__(self, qa_agent, dataset_name, save_path):
        self.qa_agent = qa_agent
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.conversation_num = 0
        self.total_image_quota = 9

    def manage_conversation(self, input_question, image_url, idx):
        self.conversation_num = 0
        if isinstance(image_url, str) and "http" in image_url[:5]:
            image_url = BytesIO(requests.get(image_url).content)

        messages = [
            {
                "role": "user",
                "content": sys_prompt_1.format(input_question),
                "images": [image_url]
            }
        ]
        current_message = messages

        thought_info = {"thoughts": [], "search": [], "sub_questions": []}

        success, idx, message, answer = self.qa_agent.ask_gpt(messages, idx)

        thought_info['thoughts'].append(self.extract_query(answer, "<Thought>\n"))

        while self.conversation_num < 5:
            if any(phrase in answer
                   for phrase in
                   ["Image Retrieval with Input Image", "Text Retrieval", "Image Retrieval with Text Query"]):
                tmp_d = {"role": "assistant"}
                tmp_d.update(message)
                current_message.append(tmp_d)
                sub_question = self.extract_query(answer, "<Sub-Question>\n")
                search_images, search_text = self.handle_retrieval(answer, image_url, idx)

                thought_info['sub_questions'].append(sub_question)
                if search_images:
                    thought_info['search'].append(search_images[0][0])
                else:
                    thought_info['search'].append("\n".join(search_text))

                contents = self.prepare_contents(search_images, messages, sub_question, idx, search_text, image_url)
                new_item = {"role": "user", "content": "\n".join(contents["text"])}
                if "images" in contents:
                    new_item.update({"images": contents["images"]})
                current_message.append(new_item)

                success, idx, message, answer = self.qa_agent.ask_gpt(current_message, idx)
                logger.info("conversation step: {} {}".format(self.conversation_num, answer))
                if not success:
                    logger.error("Request failed.")
                    break

            if "Final Answer" in answer:
                # 生成一个字典 tmp_d，代表助手的角色，并将 message 内容添加到其中。
                # 将 tmp_d 添加到 current_message 中以保留完整会话记录。
                tmp_d = {"role": "assistant"}
                tmp_d.update(message)
                current_message.append(tmp_d)
                logger.info(answer)
                logger.info("-------")
                logger.info(answer.split("Final Answer: ")[-1])
                # 返回最终答案（去掉 Final Answer: 前缀）、当前会话状态

                logger.debug(f"{thought_info=}")

                return answer.split("Final Answer:")[-1].strip(), current_message, thought_info

            logger.debug(self.conversation_num)
            self.conversation_num += 1

        logger.info(answer)
        logger.info(self.conversation_num)
        logger.info("OVER!")

        logger.debug(f"{thought_info=}")

        return answer, current_message, thought_info

    def handle_retrieval(self, answer, image_url, idx) -> Tuple[List[Tuple[str, str]], List[str]]:
        if "Image Retrieval with Input Image" in answer:
            # duckduckgo does not support reverse image search,
            # so we use multimodal model to get the image caption first, and
            # then do the search operation
            messages = [
                {
                    "role": "user",
                    "content": "What is this?",
                    "images": [image_url]
                }
            ]
            success, idx, message, answer = self.qa_agent.ask_gpt(messages, idx)

            return get_search_service().fine_search(answer,
                                                    'img_search_img',
                                                    self.save_path,
                                                    self.dataset_name,
                                                    idx,
                                                    self.conversation_num)

        elif "Text Retrieval" in answer:
            query = self.extract_query(answer, 'Text Retrieval')
            return get_search_service().fine_search(query,
                                                    'text_search_text',
                                                    self.save_path,
                                                    self.dataset_name,
                                                    idx,
                                                    self.conversation_num)

        elif "Image Retrieval with Text Query" in answer:
            query = self.extract_query(answer, 'Image Retrieval with Text Query')
            return get_search_service().fine_search(query,
                                                    'text_search_img',
                                                    self.save_path,
                                                    self.dataset_name,
                                                    idx,
                                                    self.conversation_num)

    def extract_query(self, answer, retrieval_type):
        # the query based on the given retrieval type.
        # For example, in the case of "Text Retrieval", we extract
        # the query from the answer after "Text Retrieval:".
        # In the case of "Image Retrieval with Text Query", we extract
        # the query from the answer after "Image Retrieval with Text Query:".
        # In the case of "Image Retrieval with Input Image", we extract
        # the query from the answer after "Image Retrieval with Input Image:".
        #
        # Note: This is a simplified example and may not work for all cases.
        # You may need to modify this based on the specific requirements of your project
        extract_pattern = f"(?<={retrieval_type})([\s\S]*?)(?=\n)"
        query = re.search(extract_pattern, answer, re.DOTALL)
        if not query:
            query = ''
        else:
            query = query.group(1).strip()
        logger.debug(f"{answer=}\n{retrieval_type=}\n{query=}")
        return query

    def prepare_contents(self, search_images, messages, sub_question, idx, search_text, image_url):
        if len(search_images) > 0:
            # 断言失败的时候显示(search_text)
            # assert len(search_images) == len(search_text), (search_text)
            contents = {"text": ["Contents of retrieved images: "], "images": []}
            use_imgs_num = min(5, self.total_image_quota)
            self.total_image_quota -= use_imgs_num
            for img, txt in zip(search_images[:use_imgs_num], search_text[:use_imgs_num]):
                contents["text"].extend(["Description: " + txt])

                if "http" in img[0][:5]:
                    img_item = BytesIO(requests.get(img[0]).content)
                else:
                    img_item = img[0]

                contents["images"].extend([img_item])

        else:
            contents = ["Below are related documents, which may be helpful for answering questions later on:"]
            for txt in search_text:
                contents.append(txt)
            contents.append("\nWe also provide a related image.")
            contents.append(sub_question + ' Answer:')
            contents = '\n'.join(contents)

            sub_messages = [
                {
                    "role": "user",
                    "content": contents,
                    "images": [image_url]
                }
            ]

            success = True
            _, _, _, answer = self.qa_agent.ask_gpt(sub_messages, idx)
            contents = {"text": ["Contents of retrieved documents: "]}
            if success:
                contents["text"].extend([answer])
            else:
                for txt in search_text:
                    contents["text"].extend([txt])

        return contents
