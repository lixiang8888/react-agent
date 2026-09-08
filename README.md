关于REACT agent的个人实践。

### agent架构

##### 1.agent提示词

* 定义agent的角色
* 列出agent可用的工具清单
* 严格规定thought以及action格式
* 调用过程中输入上下文本，让agent根据历史记录回答

##### 2.定义类

###### run类

* 1）格式化提示词
* 2）调用LLM思考
* 3）解析LLM的输出
* 4）执行action
* 5）将本轮的actionobservation加入历史记录
* 6）重复一定次数的第2步到第5步

###### parse-output类

分离出thought与action

###### parse-action类

 进一步解析action类

###### 工具类

 负责管理与执行工具，一个工具清单

###### 搜索工具

 使用Tavily的api
