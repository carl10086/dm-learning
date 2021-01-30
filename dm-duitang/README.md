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

## Q2: 关于 BlogForwardRepoImpl

目前的代码:

- BlogForward 这个 entity 中包含了一个叫做 BlogMeta 的东西 .
- 通过 组合2个 BlogDAO 在 RepoImpl 中完成. 通过 Converter 的组合来完成? 数据库的组装逻辑个人认为应该和业务逻辑无关 ?

## Q3: 关于 BlogDetail 的东西. 这个横跨多个系统的聚合对象 ?

- 通过 组合多个 entity 对象组合

跟 Q2 比起来就是放弃了以前在 Service 中出现了 fillXXX 的代码(这部分代码不属于 业务逻辑).

## Q4: Blog 的丰富查询实现在 哪里?

审核 查询需要 BlogMeta ; 搜索 查询需要 BlogDetail ;

感觉是一个很坑爹 东西就是 BlogDetail 没有用处 !!!!!!!!, 应该在 DTO 的逻辑中实现 RpcServiceImpl 中实现这个组合逻辑

# Reference

- [阿里 DDD 系列 1](https://developer.aliyun.com/article/716908)
- [阿里 DDD 系列 2](https://developer.aliyun.com/article/719251)
