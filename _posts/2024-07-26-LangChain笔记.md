---
date: 2024-07-26 15:11:26
layout: post
title: LangChain笔记
subtitle: 'LangChain笔记'
description: LangChain
category: 大模型
image: https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fad2da2e9-47ff-46df-87f4-59385508c935_1164x1316.png
tags:
  - LangChain
  - 大模型
  - LLM
author: 沙中世界
---

LangChain本质上是一个框架，所谓框架就是对某一门关键技术的封装，使之应用起来更加的方便；<br>
简单场景下，原生API足矣，使用框架反而可能有些多余，框架主要是对于复杂应用而言的。

LangChain是对LLM的封装，LLM（大语言模型）可以简单理解为文本对话系统，典型的例子就是chatGPT，输入一段文字描述，可以是任意的问题，然后会返回相应的回答；

复杂的框架通常对应着一系列的组件、模块，下面进行逐一介绍

### prompts模板
open AI原生的API是这样调用的

```python
  import openai
  response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Who won the world series in 2020?"},
      {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
      {"role": "user", "content": "Where was it played?"}
    ]
  )

  print(response.choices[0].message["content"])
```

LangChain对openAI的api进行了封装，通过模板的方式，使得在特定场景下，很多逻辑可以复用，提升开发效率，示例如下：

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
chat = ChatOpenAI(temperature=0.0, model=llm_model)
template_string = """Translate the text \
                  that is delimited by triple backticks \
                  into a style that is {style}. \
                  text: ```{text}```
                  """
prompt_template = ChatPromptTemplate.from_template(template_string)
customer_style = """American English \
                in a calm and respectful tone
                """
customer_email = """
                Arrr, I be fuming that me blender lid \
                flew off and splattered me kitchen walls \
                with smoothie! And to make matters worse, \
                the warranty don't cover the cost of \
                cleaning up me kitchen. I need yer help \
                right now, matey!
                """
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
customer_response = chat(customer_messages)
print(customer_response.content)
```

上面的代码看起来很长，但如果类似的逻辑需要处理很多次，就会显著减少代码量，提升效率；

prompts template本质上就是总结出典型的、通用的、好用的prompts格式，中间利用占位符来表示可变内容，这样每次使用只需要提供那些变量，复用那些通用的格式，达到高效、好用的目的；

### output parser
在prompts中指定返回的格式和内容，然后自动化解析，示例如下:
```python
# 在prompts中增加如下要求，约束返回的格式和内容
"""
Format the output as JSON with the following keys:
gift
delivery_days
price_value
"""

"""
但是LLM本质上返回的是字符串，只能约束返回json字符串，要处理这些内容，最佳的方式就是先转成json或者dict对象，然后再做信息提取和处理就会方便很多；

这个时候output parser就应运而生了，完整的代码量较多 这里截取关键的代码片段
"""
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
"""
返回内容：
{
	"gift": true,
	"delivery_days": 2,
	"price_value": ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]
}
"""
output_dict = output_parser.parse(response.content)
output_dict.get('delivery_days')
```

### memory
LLM本质上是文本输入到输出的映射，自身不具备记忆功能，但很多场景下多轮对话，是需要依赖上下文背景的，比如：
```
"""
human:      "Hi, my name is Andrew"
assistant:  "Hello Andrew! It's nice to meet you. How can I assist you today?"
human:      "What is 1+1?"
assistant:  "1+1 equals 2. Is there anything else you would like to know?"
human:      "What is my name?"
assistant:  "Your name is Andrew."
"""
```
如果没有上下文信息，最后一个问题AI是没办法回答的，LangChain提供了丰富的上下文管理方式，并且自动加入到prompts中，大幅提升开发效率；

方式一：ConversationBufferMemory

存储所有的上下文信息，更符合人类直觉认知，似乎就应该这样做；<br>
这样做也会带来一些弊端：① 上下文信息太长，模型响应可能会比较慢；② 可能超出模型最大输入限制
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "hi"}, {"output": "whats up"})

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
llm = ChatOpenAI(temperature=0) # 这里如果使用OpenAI，会报：InvalidRequestError
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)
conversation.predict(input="Hi there!")
```

方式二：ConversationBufferWindowMemory<br>
按照对话轮次记忆，通过参数设定需要记忆的对话轮次

方式三：ConversationTokenBufferMemory<br>
按照token数量记忆

方式四：ConversationSummaryBufferMemory<br>
多轮对话之后，利用LLM写一个summary，同时可以限制最大token数量；

方式五：ConversationKnowledgeGraphMemory

方式六：ConversationEntityMemory

### LLM chain
chain feature本质上是对一系列的[input → output]的封装整合；基本单元是：LLMChain<br>
一个LLMChain可以理解为一个input → output的映射；

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9, model=llm_model)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)
```
#### SimpleSequentialChain
每一个chain只有一个输入、输出；<br>
上一个chain的输出为下一个chain的输入，类似于单链表；
![SimpleSeqChain](/post_images/llm/SimpleSeqChain.png)
```python
from langchain.chains import SimpleSequentialChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )
overall_simple_chain.run(product)
```

#### SequentialChain
每一个chain可以拥有多个输入，连接前面的多个chain，但通常只有1个输出<br>
需要显式指定每个chain每一个输入、输出的名称；<br>

改造成多个输出也不难，就是两个或者多个chain，输入一样  输出不一样，相当于变相实现了多输入对应多输出；

![SeqChain](/post_images/llm/SeqChain.png)

```python
from langchain.chains import SequentialChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to chinese:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="Chinese_Review"
                    )

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence, use chinese:"
    "\n\n{Chinese_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )
# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )
# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["Chinese_Review", "summary","followup_message"],
    verbose=True
)
review = df.Review[5]
overall_chain(review)
```

#### Router Chain
思路：预先定义一系列的chain，然后根据LLM的返回值，选择chain；<br>
实现：定义router chain 和 destination_chains；router chain访问LLM，返回destination chains的name；<br>
然后根据name，选择chain
```python
chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
```
![RouterChain](/post_images/llm/RouterChain.png)
完整示例:
```python
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model=llm_model)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
\```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
\```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
chain.run("What is black body radiation?")
chain.run("what is 2 + 2")
chain.run("Why does every cell in our body contain DNA?")

```

### Question Answer
这个feature本质上属于RAG（Retrieval-Augmented Generation）技术；<br>
也就是基于一系列的文档，让LLM对问题的回答更加的准确，这些文档可能是私域内的或者是比较新的数据、信息等，不在大模型训练范围内的；<br>

通常会先将文档向量化，以便于搜索查询相关主题；<br>
调用大模型之前，会先通过向量数据库，查询到相关数据/信息，一起传送给大模型；

所谓的向量化，就是将一段文本（可能是一个句子，一个段落或者一个文档）编码成一个向量，便于查找；

如果基于原生API，流程大概是这样的；
```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed)) #1536

# 将一系列的文档编码成向量
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
# 查找最相似的向量
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

# 调用大模型
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 
```

LangChain 封装成了如下的使用方式：
```python
retriever = db.as_retriever()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
```

### 参考资料
https://learn.deeplearning.ai/courses/langchain/lesson/1/introduction <br>
https://www.openaidoc.com.cn/docs/guides/chat


