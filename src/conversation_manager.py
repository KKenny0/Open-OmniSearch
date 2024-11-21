import io
from io import BytesIO
import requests
from PIL import Image
from loguru import logger

from .prompt import *
from .search_duck import fine_search


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

        success, idx, message, answer = self.qa_agent.ask_gpt(messages, idx)
        logger.info(f"first response: {answer}")
        logger.info(f"first message: {message}")

        chain_of_thought = []

        while self.conversation_num < 5:
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
                return answer.split("Final Answer: ")[-1], current_message

            if any(phrase in answer
                   for phrase in
                   ["Image Retrieval with Input Image", "Text Retrieval", "Image Retrieval with Text Query"]):
                tmp_d = {"role": "assistant"}
                tmp_d.update(message)
                current_message.append(tmp_d)
                sub_question = answer.split('<Sub-Question>\n')[-1].split('\n')[0]
                search_images, search_text = self.handle_retrieval(answer, image_url, idx)

                contents = self.prepare_contents(search_images, messages, sub_question, idx, search_text, image_url)
                new_item = {"role": "user", "content": "\n".join(contents["text"])}
                if "images" in contents:
                    new_item.update({"images": contents["images"]})
                current_message.append(new_item)

                success, idx, message, answer = self.qa_agent.ask_gpt(current_message, idx)
                logger.debug("conversation step: {} {}".format(self.conversation_num, answer))
                if not success:
                    logger.error("Request failed.")
                    break
            logger.debug(self.conversation_num)
            self.conversation_num += 1
        logger.info(answer)
        logger.info(self.conversation_num)
        # print(current_message)
        logger.info("OVER!")
        return answer, current_message

    def handle_retrieval(self, answer, image_url, idx):
        if "Image Retrieval with Input Image" in answer:
            # duckduckgo does not support reverse image search,
            # so we use multimodal model to get the image caption first, and
            # then do the search operation

            messages = [
                {
                    "role": "user",
                    "content": "What is in this image?",
                    "images": [image_url]
                }
            ]
            success, idx, message, answer = self.qa_agent.ask_gpt(messages, idx)

            return fine_search(answer,
                               'img_search_img',
                               self.save_path,
                               self.dataset_name,
                               idx,
                               self.conversation_num)

        elif "Text Retrieval" in answer:
            query = self.extract_query(answer, 'Text Retrieval')
            return fine_search(query,
                               'text_search_text',
                               self.save_path,
                               self.dataset_name,
                               idx,
                               self.conversation_num)

        elif "Image Retrieval with Text Query" in answer:
            query = self.extract_query(answer, 'Image Retrieval with Text Query')
            return fine_search(query,
                               'text_search_img',
                               self.save_path,
                               self.dataset_name,
                               idx,
                               self.conversation_num)

    def extract_query(self, answer, retrieval_type):
        return answer.split(retrieval_type)[-1].replace(':', '').replace('"', '').replace('>', '')

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
                    img_item = Image.open(requests.get(img[0], stream=True).raw)
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
                    "images": [BytesIO(requests.get(image_url).content)] if "http" in image_url[:5] else [image_url]
                }
            ]

            success = True
            answer = self.qa_agent.ask_gpt(sub_messages, idx)
            contents = {"text": ["Contents of retrieved documents: "]}
            if success:
                contents["text"].extend([answer])
            else:
                for txt in search_text:
                    contents["text"].extend([txt])
        return contents
