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
```
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

### 参考资料
https://learn.deeplearning.ai/courses/langchain/lesson/1/introduction <br>
https://www.openaidoc.com.cn/docs/guides/chat


