## overview

- design for self healing : in a distributed system, failures happen. Design your application to be self healing when
  failures occurs.
- Make all things redundant : Build redundancy into your application, to avoid have single points of failure.
- Minimize coordination:  minimize coordination between application services to achieve scalability .
- Deisgn to scale out ...

## design for self healing

In a distributed system :

- failures can happen
- hardware can fail
- the network can have transient failures
- Rarely, an entire service or region may experience a disruption, but even those must be planned for.

Therefore, design an application to be self-healing when failures occur. This requires a three-pronged approach.

- Detect failures.
- Respond to failures gracefully .
- Log and monitor failures , to give operational insight .

Also, don't just consider big events like regional outages, which are generally rare.
You should focus as much, if not more, on handling local, short-lived failures, such as a network connectivity failures
or failed database connections .

the following is the recommendations :

- Retry failed operations. Transient failures may occur due to momentary[瞬时] loss of network connectivity , a dropped
  database connection, or a timeout when a service is busy . many client support auto retry .
- Circuit breaker . After a transient failure, but if the failure persists, you can end up with to many callers
  hammering a failing service. This can lead to cascading failures, as requests back up . Use the circuit breaker
  patterns to failed fast .
- Isolate critical resources (Bulkhead). Failures in one subsystem can sometimes cascade . This can happen if a failure
  causes some resources, such as threads or sockets, not to get freed in a timely manner, leading to resource
  exhaustion. To avoid this, partition a system into isolated groups, so that a failure in one partition does not bring
  down the entire system .
- Fail over, if an instance can't be reached, fail over to another instance. For things are stateless , like a web
  server, put several instances behind a load balancer or traffic manager . For things that store state, like a
  database, use replicas and fail over.
- Compensate failed transactions . In general, avoid distributed transactions, as they require coordination across
  services and resources. Instead, compose an operation from smaller individual transactions. If the operation fails
  midway through, use Compensating Transactions to undo any step that already completed.
- Checkpoint long-running transactions. Checkpoints can provide resiliency if a long-running operation fails. 
- Degrade gracefully. Entire subsystems might be noncritical for the application .



## refer

- [design-patterns](https://learn.microsoft.com/en-us/azure/architecture/guide/design-principles/self-healing)