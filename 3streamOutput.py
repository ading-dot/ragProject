from LlamaIndex_openai import embedding_model
from LlamaIndex_openai import llm
persist_dir = "./storage"
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from llama_index.core import StorageContext,load_index_from_storage

vector_store = FaissVectorStore.from_persist_dir(persist_dir)
storage_context = StorageContext.from_defaults(
    vector_store = vector_store,persist_dir = persist_dir
)
index = load_index_from_storage(storage_context = storage_context,embed_model = embedding_model)
print("从硬盘上加载索引：",index.docstore.docs)

#1. 构建流式输出引擎
# 后端：流式输出=》前端捕获=》聊天界面中流式输出
query_engine = index.as_query_engine(
    streaming = True,# 启用流式响应
    similarity_top_k=3,#返回最相似的3个结果
    llm=llm  #指定使用的语言模型
)
# 2.response_stream中有一个生成器，response_stream.response_gen
# response_stream = query_engine.query("请写一篇1000字的文章关于广告专业的就业前景")
# for text in response_stream.response_gen:
#     print(text,end="")#无 end 参数, 自动添加换行符（\n)
# 3.用fastapi做后端，这是一个新的python框架,用fastapi做成一个http接口
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
app = FastAPI()  #创建 FastAPI 应用实例
# 添加CORS中间件，允许所有来源跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']  #开放给所有域名
)
# 定义流式聊天接口
@app.get('/stream_chat')
async def stream_chat(param: str="你好"):
#     定义生成器函数，用于流式生成响应
    def generate():
        # 使用预先构建的 query_engine 执行查询
        response_stream = query_engine.query(param)
        # 逐段返回响应文本
        for text in response_stream.response_gen:
            yield text

    # 返回流式响应，媒体类型为 text/event-stream
    return StreamingResponse(generate(),media_type="text/event-stream")


# 主程序入口
if __name__ =='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8081)
# if __name__ == '__main__': 下的代码是同步上下文，不能直接使用 await,可以通过 asyncio.run 启动异步服务
#     启动FastAPI服务，监听所有网络接口的8081端口

# 下面的代码由于在python文件中运行，所以不用，直接用 uvicorn.run即可
#       import asyncio
#       config = uvicorn.Config(app, host='0.0.0.0', port=5000)
#       server = uvicorn.Server(config)
#       asyncio.run(server.serve())  # 通过 asyncio.run 启动异步服务
#


