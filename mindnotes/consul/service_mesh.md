## intro

A service mesh is a dedicated infra layer for handling service-to-service communication. It's responsible for the
reliable delivery of requests through the complex topology of services that comprise a modern, cloud native application
.

- 2016 william morgan oliver gould -> linkerd in
  github  [. whats-a-service-mesh-and-why-do-i-need-one.](https://linkerd.io/2017/04/25/whats-a-service-mesh-and-why-do-i-need-one/)
    - With hundreds of services or thousands of instances, and an orchestration layer that's rescheduling instances from
      moment to moment <- it may be too complex of service topology .
    - Containers make it easy for each service to be written in a different language, **the library approach is no
      longer feasible.**

This combination of complexity and criticality motivates the need for **a dedicated layer** for **service-to-service
communication** decoupled from application code and **able to capture the highly dynamic nature of the underlying env**.
This layer is `the service mesh`.

questions:

- routing
- fault-tolerant
- limit req
- encryption
- authentication
- authorization
- tracking
- metrics

## cost of communication between services

**phase 1**

Service communication treated as part of the non-functional requirements . The reliability of communication is
guaranteed by the programmer.

The developer have tools like 'OKHttp'. `Grpc`. Biz code is coupled with the communication-handler logic.

**phase 2**

A consistently, effective way for developers to decouple dependencies is to extract separate code as a couple of public
component libraries.

like `dubbo` , `spring cloud components` ...

**phase 3**

In this phase, the public component library responsible for communication is separated outside the process, and the
programs interact with each other by a network proxy .

- For high perf: In the same host vm or container . Use loop network or Unix domain socket to interactive .

**phase 4**

- the network agent is injected into the application container in the form of a side-car .
- which automatically hijacks the network traffic of the application
- the reliability of the communication is guaranteed by a dedicated communication infrastructure .

**phase 5**

service mesh .

- Data plane
- Control plane

## refer

- [使用 ebpf 代替 iptables 在 envovy 实现网络加速](http://blog.daocloud.io/7949.html)
- [merbridge - Accelerate your mesh with eBPF](https://istio.io/latest/blog/2022/merbridge/)


