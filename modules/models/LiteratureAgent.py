import logging
import textwrap
import requests
import json
import colorama
import traceback
from ..utils import *
from ..presets import *
from .base_model import BaseLLMModel
import time

class LiteratureAgent_Client(BaseLLMModel):
    # 209.97.149.43/localhost
    def __init__(self, model_name, api_key, user_name="", base_url="http://121.5.254.174:46666") -> None:
        super().__init__(model_name=model_name, user=user_name, config={"api_key": api_key})
        self.base_url = base_url
        print("self.base_url",self.base_url)
        self.session_id = None
        self.history = []
        self.system_prompt = """You are a literature agent that helps users analyze and understand paper works. 
        You can provide insights on papers"""

    def _get_literature_style_input(self):
        messages = []
        for item in self.history:
            if item["role"] == "user":
                messages.append({"role": "user", "content": item["content"]})
            else:
                messages.append({"role": "assistant", "content": item["content"]})
        return messages

    def _make_api_call(self, messages, stream=False):
        print("_make_api_call stream", stream)
        url = f"{self.base_url}/v1/chat/completions"
        print("_make_api_call messages", messages)
        print('self.session_id', self.session_id)
        print('stream', stream)

        if self.session_id:
            url = f"{url}/{self.session_id}/continue"
            payload = messages[-1]  # 只发送最后一条消息用于继续对话
        else:
            payload = {
                "messages": messages,
                "stream": stream,
                "user_id": None,
                "session_id": None
            }

        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            if stream:
                if response.status_code == 200:
                    return response
                else:
                    logging.error(f"Stream API call failed with status code: {response.status_code}")
                    return None
                
            data = response.json()
            
            if not self.session_id:
                self.session_id = data.get("id", "").replace("lit-", "")
            
            end_time = time.time()
            cost_time = end_time - start_time
            print(f"API call took {cost_time:.2f} seconds")
            
            # 处理非流式响应
            res = data.get("content") or data.get("response") or data.get("choices")[0]['message']['content']
            if isinstance(res, dict) and "content" in res:
                res = res["content"]
            return res
            
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            return None

    def get_answer_at_once(self):
        try:
            messages = self._get_literature_style_input()
            response = self._make_api_call(messages)
            if response:
                return response, len(response)
            else:
                return "Unable to generate response", 0
        except Exception as e:
            logging.error(f"Error in LiteratureAgent: {str(e)}")
            return f"An error occurred: {str(e)}", 0

    def get_answer_stream_iter(self):
        messages = self._get_literature_style_input()
        partial_text = ""
        try:
            response = self._make_api_call(messages, stream=True)
            if response:
                # 处理SSE流式响应
                for line in response.iter_lines():
                    if line:
                        try:
                            # 解析 SSE 数据
                            data = json.loads(line.decode('utf-8'))
                            if 'choices' in data and len(data['choices']) > 0:
                                if 'delta' in data['choices'][0]:
                                    content = data['choices'][0]['delta'].get('content', '')
                                else:
                                    content = data['choices'][0]['message'].get('content', '')
                                
                                if content:
                                    partial_text += content
                                    yield partial_text
                        except json.JSONDecodeError:
                            continue
            else:
                yield "Unable to generate response"
        except Exception as e:
            logging.error(f"Error in streaming response: {str(e)}")
            yield f"An error occurred: {str(e)}"

    def to_markdown(self, text):
        text = text.replace("•", "  *")
        return textwrap.indent(text, "> ", predicate=lambda _: True)

    def handle_file_upload(self, files, chatbot, language):
        if files:
            try:
                for file in files:
                    if file.name:
                        with open(file.name, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.history.append({
                                "role": "user", 
                                "content": f"Please analyze this text:\n\n{content}"
                            })
                            chatbot = chatbot + [((file.name,), None)]
                return None, chatbot, None
            except Exception as e:
                logging.error(f"Error handling file upload: {str(e)}")
                return f"Error processing file: {str(e)}", chatbot, None
        return None, chatbot, None

    def predict(
        self,
        inputs,
        chatbot,
        use_websearch=False,
        files=None,
        reply_language="中文",
        should_check_token_count=True
    ):
        status_text = "开始生成回答……"
        if type(inputs) == list:
            logging.info(
                "用户"
                + f"{self.user_name}"
                + "的输入为："
                + colorama.Fore.BLUE
                + "("
                + str(len(inputs) - 1)
                + " images) "
                + f"{inputs[0]['text']}"
                + colorama.Style.RESET_ALL
            )
        else:
            logging.info(
                "用户"
                + f"{self.user_name}"
                + "的输入为："
                + colorama.Fore.BLUE
                + f"{inputs}"
                + colorama.Style.RESET_ALL
            )

        # 添加用户输入到历史记录
        if type(inputs) == list:
            self.history.append(inputs)
        else:
            self.history.append({"role": "user", "content": inputs})

        # 准备显示
        if type(inputs) == list:
            display_input = inputs[0]["text"]
        else:
            display_input = inputs
        chatbot.append((display_input, ""))
        yield chatbot, status_text

        try:
            # 使用 LiteratureAgent 的 API 调用逻辑
            messages = self._get_literature_style_input()
            response = self._make_api_call(messages, stream=self.stream)
            print("response",response)
            
            if self.stream:
                if response is None:
                    error_message = "Failed to get streaming response from API"
                    chatbot[-1] = (display_input, error_message)
                    yield chatbot, status_text
                else:
                    partial_text = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'choices' in data and len(data['choices']) > 0:
                                    if 'delta' in data['choices'][0]:
                                        content = data['choices'][0]['delta'].get('content', '')
                                    else:
                                        content = data['choices'][0]['message'].get('content', '')
                                    
                                    if content:
                                        partial_text += content
                                        chatbot[-1] = (display_input, partial_text)
                                        yield chatbot, status_text
                            except json.JSONDecodeError:
                                continue
            else:
                # 从响应对象中提取实际的内容文本
                response_text = response.get('content') if isinstance(response, dict) else response
                chatbot[-1] = (display_input, response_text)
                yield chatbot, status_text

            # 添加助手回复到历史记录
            self.history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            traceback.print_exc()
            status_text = STANDARD_ERROR_MSG + beautify_err_msg(str(e))
            yield chatbot, status_text

        self.chatbot = chatbot
        self.auto_save(chatbot)