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

### 参考资料
https://learn.deeplearning.ai/courses/langchain/lesson/1/introduction <br>
https://www.openaidoc.com.cn/docs/guides/chat


