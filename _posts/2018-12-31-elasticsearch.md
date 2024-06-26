---
date: 2018-12-31 21:44:40
layout: post
title: elasticsearch
description: elasticsearch
image: https://p3-tt.byteimg.com/origin/pgc-image/af23f692d089424ba5e81b6bf0126300
optimized_image: https://p3-tt.byteimg.com/origin/pgc-image/af23f692d089424ba5e81b6bf0126300
category: 大数据
tags:
  - 大数据
  - elasticsearch
  - 搜索
author: 沙中世界
---

分布式全文搜索引擎，主要功能是对大数据的存储以及近似实时的搜索与分析功能

### 数据存储方式
面向文档的存储方式，并且对内容进行索引。<br>

#### 索引与文档
**索引**：作为名词时，可以类比于关系型数据库中的表。作为动词时就是将文档写入索引中(包括倒排索引与Doc value的建立)，便于被检索的过程。

**文档**：就是一个记录，可以用于存储非结构化数据

#### 数据存储结构

**倒排索引**：<br>
将文档内容拆成词条(Terms/tokens)，然后创建一个包含所有词条的的排序列表，然后将这个词条与之对应的一个或者多个文档做映射。由词条到文档的映射。

Term| doc1 | doc2
--|--|--
term1|√ |
term2| |√
term3|√|√
...

倒排索引时可以快速搜索的重要原因之一。

**doc values**：<br>
完成文档到词项的映射，在索引数据与建立倒排索引时同时建立doc values。主要用于排序，聚合。

除了analyzed string字段之外，其他字段默认都会记录doc value。doc values的存储使用了一些压缩机制。对于数字类型字段，一般取最大公约数，然后只记录其倍数。如果没有合适的最大公约数，取最小的数值，然后记录其差值。

如果某个字段不需要这个功能呢，可以通过配置字段的属性:"doc_values": false 来禁用。

**Lucene存储**<br>
Lucene是一个功能强大的搜索库，但是基于Lucene进行开发比较复杂。ElasticSearch是基于lucene开发的搜索引擎，提供了更简单易用的API。

索引实际上是lucene中的概念，一个索引由多个索引段构成，大部分的场景是写一次，读多次。当满足某些条件时，多个索引段会合并成一个更大的索引段。索引段的减少有助于搜索效率的提高(可能是lucene内部原理决定的)，但是频繁的段合并会影响性能。

Elasticsearch中的每次刷新都会新创建一个段，新创建的段里面的数据在下一次刷新之前是不会被搜索到的。ES的段合并是在后台进行的。

#### 分布式与容错机制
**路由分片**：根据如下公式计算该文档所在的分片id，然后放入到相应分片中。
> shard = hash(routing) % number_of_primary_shards routing默认是文档的id，也可以自定义，跟文档有关的api都会接受一个routing的参数，通过这个参数自定义文档到分片的映射。<br>
这就解释了在创建索引时就定义好分片数量，一旦确定好后就不能再改变，否则之前的路由就会失效。

**分片容错**：每一个逻辑分片会对应主分片和副本分片。当某台机器宕机之后，超过一定时间(优化参数之一)，ES就会按照配置，重新规划分片的位置。

### 搜索功能
默认的，文档中的每一个属性都是会被索引和可搜索的。
- 根据记录id进行搜索
- 针对某字段进行精确值搜索(term/terms关键字)
- 对数值及日期字段进行区间搜索
- 针对文本字段的相关搜索(使用match关键字)，并返回相关性得分<br>
这里会涉及到一些机器学习与自然语言处理的知识，简单举一个例子：搜索“攀岩”，会得到攀岩、爬山等结果。具体原理后续系列详细介绍。

- 搜索文本中的短语(使用match_phrase关键字)
- 支持高亮搜索，即将搜索结果中被搜索的字段进行高亮标识
- 支持分页搜索和TopN搜索
- 多个搜索条件可以进行或、且、非组合

特别说明，对于null值的特殊处理，如果搜索某个字段值等于null的文档，首先应该使用exist判断该文档存在这个字段，然后再搜索。

**分析器**<br>
由字符过滤器，分词器，词单元过滤器三个函数构成的一个包装类，这三个函数按照顺序执行。<br>
字符过滤器用于过滤掉那些不想被索引的词。如：html标签等<br>
把字符串分解成单个词条或者词汇单元。<br>
词单元过滤器：经过分词，作为结果的 词单元流 会按照指定的顺序通过指定的词单元过滤器 。词单元过滤器可以修改、添加或者移除词单元，如大小写统一，去掉停用词等。

测试分析器：

**自定义分词器**：


### 统计分析

**桶：**

**指标：**


### 集群相关
**节点：** 一个运行的elasticsearch实例，就是一个节点。<br>
**集群：** 一个或者多个拥有相同cluster.name的节点构成了集群。<br>
特别说明一下，对于单台机器上的实例，如果cluster.name相同，会自动组成集群(单播模式)<br>
单播模式下，不同机器上的节点不会组成集群。与单播模式相对应的是组播模式，组播模式下不同机器上的节点满足一定条件也会自动发现并组成集群的机制。<br>
单播模式相比组播模式更加安全，默认是单播模式，组播模式下可能会出现某些节点意外加入集群。<br>
**节点类型：**<br>
主节点：管理集群范围内的变更，如节点的加入、删除；索引的增加删除等；但不需要涉及文档级别的变更及搜索，所以即使流量增加，数据增加主节点一般也不是瓶颈。

数据节点：数据节点上保存了数据分片，主要负责数据相关的操作，如文档的增删改查等。

路由节点：既不负责集群内的一些变更，也不负责数据的存储，其主要作用是转发请求，起到负载均衡的作用。

摄取节点（ingest node）：在索引数据之前，执行一些预处理操作(需要用户配置这些操作)。

#### 选主机制
**什么时候选主：** 集群启动、master失效

**Bully选举算法：** 每个节点有一个惟一的id，对id进行排序，任何时刻当前leader都是id最大的节点。这种选举算法优点是易于实现，缺陷是但最大id的节点不稳定时，容易造成频繁的切换主节点。

**ES的改进：** 每个节点投票给自己所知的id最小的节点。某个节点得票超过一定的数目(半数以上，可配置)，并且自己也投票给自己，才可以被选举为主节点(解决脑裂问题)。为了解决主节点抖动问题，采取了一定的推迟选举机制，也就是说认为主节点失败而不是抖动的条件下，才进行选举。

**具体选举流程如下:**
![选主流程图](/my_docs/bigData/images/4-1.jpg)

**选主之后如何同步数据：** 新选的master未必有最新的集群状态信息，新的master会枚举集群中所有有资格称为master的节点的状态信息，然后通过版本号选取最新的状态信息作为当前集群的状态信息。

#### 集群扩容
新节点的加入，不需要修改其他节点的列表。

原理：新节点主动联系其他节点，然后得到整个集群的状态，然后找到master节点，并加入集群。

#### 宕机之后
对于普通的数据节点宕机之后，数据需要重新的rebalance


### 优化

#### 集群配置优化
- 配置服务器open file的最大数量（使用ulimit -a  查看）
- 配置启动内存，修改bin/elasticsearch 文件，增加 ES_HEAP_SIZE=4g（最大不可超过32G）
- 配置 禁止物理内存交换  config/elasticsearch.yml   bootstrap.memory_lock: true
- 禁用监控  marvel.agent.enabled（很耗CPU）
- elasticsearch.yml文件，写与读的线程池的配置
>#---------------------------------thread pool-----------------------------------<br>
　　　　threadpool.index.type: fixed <br>
　　　　thread_pool.index.size: 500 <br>
　　　　thread_pool.index.queue_size: 2000 <br>
　　　　threadpool.bulk.type: fixed <br>
　　　　threadpool.bulk.size: 100 <br>
　　　　threadpool.bulk.queue_size: 500 <br>
    
- 各司其职，配置只作为master或者data的节点，还可以配置客户端节点
#### 搜索优化
- 调整分片数目
- 能用过滤语境的，尽量不用搜索语境
- 增加刷新时间
- 减小后台merge的线程数目
- 合理的配置缓存

#### 索引优化
- 修改分片和副本的数量，太大太小都不合适（index.number_of_shards ）
- 定时对索引进行合并优化 \_forcemerge接口
- 删除已标记为删除的文档：curl -XPOST localhost:9200/uploaddata/\_forcemerge?max_num_segments=1  <br>
　　　　curl -XPOST localhost:9200/uploaddata/\_forcemerge?only_expunge_deletes=true
- 设置存储压缩方式，在速度与存储空间之间平衡(index.codec)  
- 设置刷新时间间隔 index.refresh_interval，时间增长可以增加索引速度
- 设置日志策略index.translog.durability，降低数据flush到磁盘的频率。如果对数据丢失有一定的容忍，可以打开async模式
- 宕机之后，设置分片重分配时间index.unassigned.node_left.delayed_timeout
- 后台merge的线程数 index.merge.scheduler.max_thread_count merge
- 每台机器上的分片数量index.routing.allocation.total_shards_per_node(注意，不可设置为：（ pri_shard_num + rep_shard_num) / data_node_num)
- 对于经常有取topN的需求，可设置按照某字段排序，避免全数据扫描：
>"settings" : {
        "index" : {
                    "sort.field" : "timestamp",
                    "sort.order" : "desc"
        }
    }    
    
- 对于不需要聚合的字段，可以设置doc_value为false，节省内存空间
- 对于不需要搜索的字段，可以设置index 为false
- 某个字段既不需要搜索，也不需要聚合，设置enabled=false
- 对于不需要计算评分的字段，可以设置norms为false
- 对于某个文档某个字段有异常值，但在该字段异常是，又需要保留其他字段可设置该字段ignore_malformed为true
- 对于不需要获取原始数据，只需要排序或者计算的情况下，将_source设置为false
- 必要时可以按照字段，将一个索引拆分成多个索引(对于可枚举的字段)
- 可以根据使用场景自定义分片规则(使用路由规则)
- 避免不平衡分片

### kibana使用
#### 搜索
条件连接符:AND OR NOT <br>
数字区间搜索：count: >300 AND count: <1000    其中count时字段名称

#### 统计

