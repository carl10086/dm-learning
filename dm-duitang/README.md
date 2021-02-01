# Dt blog Design

# QUESTION ?

## Q1 biz & support: 不是个好问题、因为如果是 真的服务化就解决了 .

biz support 不能够清晰的反映出依赖关系 . 或许通过 config 可以约束 ?

先解释为什么清晰的反映、 假设模块的依赖关系是如下:

- feed:
    - blog:
        - oss:
        - user:
        - counter
        - like
    - counter
    - user
    - like

解释下: feed 依赖 blog, blog 依赖 oss, blog 依赖 user . 此时 :

- blog 是防到 biz 还是 support ?
    - 如果放到 biz: 如何表达 feed 和 blog 的依赖关系
    - 如果放到 support: 如何表达 blog 和 user 的依赖关系

**Linux 系统设计中有个有意思的东西 , Linux Systemd Service 的想法 . Service 之间也会依赖**

## Q2: 关于 BlogForwardRepoImp**l

目前的代码:

- BlogForward 这个 entity 中包含了一个叫做 BlogMeta 的东西 .
- 通过 组合2个 BlogDAO 在 RepoImpl 中完成. 通过 Converter 的组合来完成? 数据库的组装逻辑个人认为应该和业务逻辑无关 ?

## Q3: 关于 BlogDetail 的东西. 这个横跨多个系统的聚合对象 ?

- 通过 组合多个 entity 对象组合

跟 Q2 比起来就是放弃了以前在 Service 中出现了 fillXXX 的代码(这部分代码不属于 业务逻辑).

## Q4: Blog 的丰富查询实现在 哪里?

[BlogDetail 应该是个错误的设计](https://blog.csdn.net/FS1360472174/article/details/88542163)

审核 查询需要 BlogMeta ; 搜索 查询需要 BlogDetail ;

感觉是一个很坑爹 东西就是 BlogDetail 没有用处 !!!!!!!!, 应该在 DTO 的逻辑中实现 RpcServiceImpl 中实现这个组合逻辑


## Q5: Domain 中是不是有分层, DomainService 是什么 ?

个人认为 domain 中分层意义不大; 分层容易、保持很难;

- [阿里 DDD 系列 1](https://developer.aliyun.com/article/716908) 中的 ExchangeRate 对象就是一个 `DomainService` 吗?
    - 不是,按照 `DomainService` 的定义 , 优秀的 DomainService 有一些特征:
        - 不是 ENTITY 或者 VALUE Object 的一个自然组成部分;
        - 接口是根据领域模型的其他元素定义的
        - 操作是无状态的 ... 这里的无状态很难理解啊 ..
- 但是 原书中用 `DomainService` 实现了转账? 多个对象交互
    - `DomainService` 应该用 动词而不是名词命名、**很关键**
    

## Q6 DomainService 的无状态是什么意思、里面可以用 Repo 吗?
    
    - 个人认为不可以用 Repo, 才能更好的无状态
    - DomainService 在实践中、应该是一个接口、实现类也应该在 Domain 中、 用 接口方法可以很好的表示 领域 Service 解决的服务是什么? 
    - 多个对象
    
    



# Reference

- [阿里 DDD 系列 1](https://developer.aliyun.com/article/716908)
- [阿里 DDD 系列 2](https://developer.aliyun.com/article/719251)
- [阿里 DDD 系列 3](https://juejin.cn/post/6845166890554228744#heading-12)




- [meituan_ddd_rumen]
- [ddd_in_practice](https://www.cnblogs.com/xiandnc/p/11070470.html#_caption_5)
- [azurl_ddd](https://docs.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/apply-simplified-microservice-cqrs-ddd-patterns)