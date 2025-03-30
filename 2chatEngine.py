# 使用lama-index来构建问答系统，包含读取文档》构建节点=》构建索引》保存和记载索引
# 首先从指定文件中读取，输入为列表
from llama_index.core import SimpleDirectoryReader,Document
documents = SimpleDirectoryReader(input_files=['./docs/问答手册.txt']).load_data()
with open('load_file.txt','w') as f:
    f.write(str(documents))
# print(documents)
# 方法一：直接使用documents构建索引
# 构建向量索引
# from llama_index.core import VectorStoreIndex
# from LlamaIndex_openai import embedding_model
# index = VectorStoreIndex.from_documents(documents,embed_model = embedding_model,show_progress = True)
# print(index)
# 方法二:先构建节点,在构建索引，同时使用faiss作为向量数据库
# SentenceSplitter:用于将文档分割成较小的文本块（节点）
# chunk_size:每个文本块的最大长度
from llama_index.core.node_parser import SentenceSplitter
# 初始化分割函数
transformations = [SentenceSplitter(chunk_size = 512)]
# print(transformations)
# run_transformations: 对文档应用一系列转换操作（如分割文本）。
# nodes: 返回分割后的文本块（节点）。
from llama_index.core.ingestion.pipeline import run_transformations
nodes = run_transformations(documents,transformations=transformations)
# print(nodes)
# 根据节点构建索引
# FaissVectorStore: 使用 Faiss 作为向量存储库。
# faiss.IndexFlatL2(dimensions) 创建一个 L2 距离的 Faiss 索引，dimensions 是嵌入向量的维度。
# vector_store:初始化 Faiss 向量存储。
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core import StorageContext,VectorStoreIndex
from LlamaIndex_openai import embeddings
dimensions = len(embeddings)
vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dimensions))
print(type(vector_store))
#用于管理存储上下文的类：StorageContext
# from_defaults(vector_store = vector_store):使用默认配置初始化存储上下文，并指定向量存储

storage_context = StorageContext.from_defaults(vector_store = vector_store)
print(type(storage_context.vector_store))

# 2.1VectorStoreIndex:从节点构建向量索引，并指定嵌入模型和存储上下文
from LlamaIndex_openai import embedding_model
# index = VectorStoreIndex(
#     nodes =nodes,
#     storage_context = storage_context,
#     embed_model = embedding_model
# )
# 2.2 保存索引到硬盘,以后就不用重复构建，直接从硬盘读取。
# persist_dir: 指定索引保存的目录。
# persist(persist_dir): 将索引保存到指定目录。
persist_dir = "./storage"
# index.storage_context.persist(persist_dir)
# # print("保存索引：",index.docstore.docs) 和下一个生成的一样
# print(type(index.storage_context.vector_store))  # 应该是 <class 'llama_index.vector_stores.faiss.FaissVectorStore'>

# 2.3 从硬盘加载索引
# from_persist_dir(persist_dir):从指定目录加载 Faiss 向量存储。
# load_index_from_storage:从存储上下文中加载索引

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core import StorageContext,load_index_from_storage

vector_store = FaissVectorStore.from_persist_dir(persist_dir)
storage_context = StorageContext.from_defaults(
    vector_store = vector_store,persist_dir = persist_dir
)
index = load_index_from_storage(storage_context = storage_context,embed_model = embedding_model)
print("从硬盘上加载索引：",index.docstore.docs)
# 2.4 使用索引进行问答
# as_query_engine(llm = llm ):将索引转换为问答引擎，llm是用于生成答案的语言模型
# query("专利申请如何收费")
# from LlamaIndex_openai import llm
# query_engine = index.as_query_engine(llm = llm)
# response = query_engine.query("生成该文章的主题？")
# print(response)

# 2.5针对faiss向量存储的管理方式
# 查看index下面的所有文档
# 查看index下面所有有ref的文档信息
# 查看index下面的所有node的id
print(index.index_struct.nodes_dict)  #一个
# 查看index下面所有有ref的文档的信息
print(index.ref_doc_info)
# 查看任意给定id的node详细信息,这个没有查到，id信息一直在变化，不生成索引节点，直接从硬盘中加载就好了
print(index.docstore.get_node('49e239bc-ba99-437b-a07f-e0b09f4a8901'))
# print(index.docstore.docs['1ac029ce-2fdd-4f60-b534-2edf521d3839'])
# 查看所有节点 ID（示例代码，依赖具体库）
# all_node_ids = list(index.docstore.docs.keys())  # llama-index 的 docstore
# print("Existing Node IDs:", all_node_ids[:5])    # 打印前 5 个 ID 示例
# 新增节点,doc_single必须是一个TextNode对象
# index.insert_nodes([doc_single])
from llama_index.core.schema import TextNode
nodes=[
    TextNode(
        text="The Shawshank Redemption",
        matadata={
            "author":"Stephen King",
            "theme":"Friendship",
            "year":1994,

        }
    ),
    TextNode(
        text="The Apple",
        matadata={
            "author": "Amy",
            "theme": "relation",
            "year": 1992,

        }
    )
]
index.insert_nodes(nodes)
all_node_ids = list(index.docstore.docs.keys())  # llama-index 的 docstore
print("Existing Node IDs:", all_node_ids[:5])    # 打印前 5 个 ID 示例




