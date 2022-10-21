## intro

**1) motivations**

an l7 proxy and **communication bus** designed for large modern service oriented architectures.

**2) Out of process architecture**

The project was born out of the belief that:  The network should be transparent to applications. When network and
application problems do occur it should be easy to determine the source of the problem.

Out of process architecture: Envovy is a self contained process that is designed to run alongside **every application
server**.

All of the Envoys form a *transparent communication mesh* <- in which each application sends and receive messages to and
from localhost and is unaware of the network topology.

two benefits over traditional library approache to `service to service` communication:

- Envoy works with any application language. Envoy transparently bridges the gap.
- Envoy can be deployed and upgrade quickly across an entire infrastructure transparently .

**3) L3/L4 filter arch**

At its core, Envoy is an L3/L4 network proxy . A pluggable filter chain mechanism -> s ...

**4) L7 http architecture**

An additional hTTP l7 filter layer . Http filters can be plugged into the HTTP connection management subsystem that
perform different tasks such as :

- [buffering](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/buffer_filter#config-http-filters-buffer)
- [global rate limiting](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting#arch-overview-global-rate-limit)
- [http routing / forwarding](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/http_routing#arch-overview-http-routing)

**5) First class HTTP/2 support**

The recommended ways for service to service communication is use HTTP/2 between all Envoys to create a mesh of
persistent connections that requests and responses can be multiplexed over.

**6) HTTP/3 support**

As of 1.19.0, Envoy now support HTTP/3 upstream and downstream .

**7) Http L7 routing**

when operation in HTTP mode, Envoy
supports [routing](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/http_routing#arch-overview-http-routing)

**8) gRPC support**

supports grpc WITH http/2 .

**9) Service discovery and dynamic configuration**

**10) Health checking**

...

## Terminology

**1) Host**

an entity capable of network communication (application on a mobile phone, server, etc...)
a physical piece of hardware could possibly have multiple hosts running on it .

**2) Downstream**

connects to envoy, send reqs and receives resp.

**3) Upstream**

receive connections and requests from Envoy and returns resps .

**4) Listener**

**5) Cluster**

A cluster is a group of logically similar upstream hosts that envoy connects to .

- service discovery
- active health checking
- load balancing policy

**6) mesh**

an "Envoy mesh" is a group of Envoy proxies that form a message passing substrate for a distributed system comprised of
many ....

**7) runtime configuration**

Out of band realtime configuration system deployed alongside Envoy.

## refer

- [what-is-envovy](https://www.envoyproxy.io/docs/envoy/latest/intro/what_is_envoy)