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

原书中对 DomainService 的介绍: 本质上是为了解决多个领域对象的协作问题，比如说转账中2个 Account 的各自+ - ;

如果放到 DomainService 中、代码大概下面样子:

```java
    public class TransferDomainService {
      private AccountRepo accountRepo;
      
      public void transfer (long fromAccountId, long toAccountId) {
        Long fromAccount = accountRepo.findById(fromAccountId);
        Long toAccount = accountRepo.findById(toAccountId);
        
        // .... 核心业务逻辑
        bizBetweenAccount(fromAccount, toAccount);
        
        accountRepo.save(fromAccount);
        accountRepo.save(toAccount);
      }
    }
```


可以看出来 TransferDomainService 中跟 Repo 相关的逻辑全部可以放到 AppSrv 中，个人认为应该放到 AppSrv. 

DomainService 为了解决的问题就是 AppSrv 和 Domain Model 都不好解决的，一个动词的行为涉及到多个对象;

放到 DomainService 意味着 核心业务逻辑不好测了，要测试 bizBetweenAccount . 必须要 mock AccountRepo . 失去了一个 ddd 的大优点、容易测试 . 


### Q7  AppSrv 为什么 分为3个?

- WriteSrv: 主要复制写操作, 写路径强制走 聚合根 -> Repo -> 修改操作
    - 可以修改聚合根中的部分实体 ;
- ReadSrv: 读模型为什么要分开、读的模型可能不一样，走 `representation` 作为 ReadModel ;
- AdmReadSrv: Adm 的写操作可以走 WriteSrv, AdmRead 操作区分的原因是 不希望 adm 的查询操作影响缓存的 冷热分布 ; 



### Q8 Factory 模式

以下内容来自于官方:  

- 装配对象的复杂工作要和对象要执行的工作分开;
- 不能把对象装配的工作交给客户端、那样更烂;

**对象的创建本身可以是一个主要操作、但是被创建的对象不适合承担复杂的装配操作, 将这些职责混在一起可能会产生难以理解的拙劣设计; 让客户端负责这个职责又会让客户端的设计引入混乱** 

任何一种面向对象的语言都提供了创建对象的方式、 例如 Java 和 C++ 的构造函数， SmallTalk 中创建实例的类方法； 但是依旧需要一种更加抽象 而且不和任何对象耦合的构造机制， 因为复杂对象的创建属于 领域层的职责， 但是这项任务又不属于那些用来表示那些用于表示模型的对象 . 

- 个人疑问、聚合根的构造器? 在构建 Blog 的时候、需要构建 Entity ValueObject 再装配起来、组成 一个领域的聚合根是不是就是指这个场景 ; 


书上建议用 Factory 模式来解决这个问题 ;  而好的工厂 有2个基本要求:

- 每个创建方法都是原子的:
    - 所有的值都必须初始化为一个正确的值
    - 构造器中一般会 有校验请求、如果校验失败或者任何的失败、要保证抛出异常、在方法签名中声明
- Factory 应该被抽象为所需要的类型、而不是要创建的具体类、Factory 可以创建接口类型? 



首先、大多数的时候构造器就够用了， 不要在构造函数中 调用其他类的构造函数; 


网上一些把领域服务和 [工厂搞到一起的例子](https://zhuanlan.zhihu.com/p/109048532) 个人感觉有问题、不科学、 领域对象解决的多个实体的业务、甚至是多个聚合根的 业务交互问题;
而 factory 解决的是复杂对象的装配问题、 通篇没有说 领域服务什么鬼、而是强调:
1. 什么场景下使用 factory ? factory 解决的问题
2. factory 的位置应该在哪里?  一个专门的对象 还在 聚合根上 ? 
3. 好的 factory 的特征;
4. 什么时候用 构造器的设计 ?
5. factory 是否使用接口? 接口要怎么设计 ? 
6. entity factory 和 valueObj factory 的区别在于 Id 的生成 .  
7. 从是否初次构建的角度 有重建这种特殊情况 ...



所以回到系统本身?

- Blog 的 publish 和 forward 2种情况? 是否需要工厂 ?
- 能不能简单使用构造器 ?
- 使用工厂的话放在哪里? 放在聚合根、 还是单独的 factory 接口 ?  factory 接口又要怎么设计


## Q9 When to use DomainService 



[repo 放在 applicationService 还是 domainService?](https://softwareengineering.stackexchange.com/questions/330428/ddd-repositories-in-application-or-domain-service)
[什么时候用domainService?](https://enterprisecraftsmanship.com/posts/domain-vs-application-services/)


# Reference

- [阿里 DDD 系列 1](https://developer.aliyun.com/article/716908)
- [阿里 DDD 系列 2](https://developer.aliyun.com/article/719251)
- [阿里 DDD 系列 3](https://juejin.cn/post/6845166890554228744#heading-12)




- [meituan_ddd_rumen]
- [ddd_in_practice](https://www.cnblogs.com/xiandnc/p/11070470.html#_caption_5)
- [azurl_ddd](https://docs.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/apply-simplified-microservice-cqrs-ddd-patterns)