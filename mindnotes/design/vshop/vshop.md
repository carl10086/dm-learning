## code style

- 如何尽量代码和架构质量的优化 ?
    - 说服别人的过程 .
    - 能落地为 Document
        - 有清晰的架构
    - 测试率到 ->

```java
public class OrderApplication {

  public void createOrder(CheckOrderCmd cmd) {
    //1. check order
    checkCreateOrder(cmd);

    //2. 
    Order order = doCreateOrder(cmd);
    // -OrderRepo

    //3. 
    Payment payment = createPayment();
    // - PaymentRepo

    //4. connect order and snapshot
    connectOrderAndPayment();
    //-OrderRepo
    //- PaymentRepo
  }
}
```

```java
public class InventoryApplicationService {

}

public class CreateInventoryApplicationService {

}
```


pkg design:

- `application`: something like `refreshContext()`
- `repo` : -> a couple of entity with an rootEntity
    - dao: decouple with `repo`
    - `must be interface`
- `domain`:
- `service`:
    - may aggregate multiple domain .
- `infra`:


## tasks:

### 理顺大多数操作的主要流程

- 每个操作的主要流程
- 商品:
    - 创建商品
    - 商品类目
    - sku
- 订单
  - 创建订单
- 商品派发
- 支付网关的流程
- 订阅

### 对细节进行设计和探讨

- 要出这个版本你考虑的基本细节
- 这个阶段要出 数据库设计
- 要出领域模型
- ..

### 代码落地

- 细节上的反复修正, 如果需要修改主要流程, 需要反向沟通.

### 重复上述过程 <- v2