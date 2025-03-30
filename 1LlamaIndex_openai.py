from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import LLMMetadata,MessageRole
import os
# 从环境变量中读取api_key
api_key = os.getenv('DASHSCOPE_API_KEY')
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
chat_model = "qwen-plus"
emb_model = "text-embedding-v3"
# class NewOpenAI(OpenAI):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     @property
#     def metadata(self) -> LLMMetadata:
#         # 创建一个新的LLMMetadata实例，只修改context_window
#         return LLMMetadata(
#             context_window=8192,
#             num_output=self.max_tokens or -1,
#             is_chat_model=True,
#             is_function_calling_model=True,
#             model_name=self.model,
#             system_role=MessageRole.USER,
#         )
#
# llm = NewOpenAI(
#     temperature = 0.95,
#     api_key = api_key,
#     model = chat_model,
#     api_base = base_url
# )

from llama_index.core.base.embeddings.base import BaseEmbedding



from llama_index.embeddings.openai import OpenAIEmbedding

# class NewOpenAIEmbedding(OpenAIEmbedding):
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#
#     @property
#     def metadata(self)->BaseEmbedding:
#         # 返回一个简单的字典或 None
#         return BaseEmbedding(
#             model_name = self.model,
#         )
#
#     @staticmethod
#     def get_engine(mode, model, _QUERY_MODE_MODEL_DICT):
#         # 绕过模型验证，直接返回默认值
#         return "default_engine"
#     配置嵌入模型
llm = OpenAI(
    temperature = 0.95,
    api_key = api_key,
    model = chat_model,
    api_base = base_url
)
# from llama_index.embeddings.openai import OpenAIEmbedding
embedding_model = OpenAIEmbedding(
# embedding_model = NewOpenAIEmbedding(
    api_key = api_key,
    model = emb_model,
    api_base = base_url  #API 的基础 URL

)

response = llm.complete("你是谁？")
# print(response)
#
embeddings = embedding_model.get_text_embedding("This is the first text.")
# print(len(embeddings), type(embeddings))
# import llama_index.embeddings.openai
# print(llama_index.embeddings.openai.__file__)
