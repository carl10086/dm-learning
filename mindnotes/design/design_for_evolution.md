## intro

an evolutionary design is key for continuous innovation .

All success applications change over time:

- whether to fix bugs
- add new features
- bring in new technologies
- or make existing systems more scalable and resilient.

If all the parts of an application are tightly coupled, it becomes very hard to introduce changes into the system. A
Change in one part of the application may break another part, or cause changes to ripper through the entire database.

This problem is not limited to monolithic applications .

microservices are becoming popular way to achieve an evolutionary design, because they address many of the
considerations listed here .

## rules

**1) Enforce high cohesion and loose coupling**

- A service is cohesive if it provides functionally that logically belongs together.
  Services are loosely coupled if you can change one service without change another .
- If you find that updating a service requires coordinated updates to other services, it may be a sign that your
  services are not cohesive.

**2) Encapsulate domain knowledge**

- When a client consumes a service, the responsibility for enforcing the business rules of the domain should not fall on
  the client.

Otherwise, every client has to enforce the business rules , and you end up with domain knowledge spread across different
parts of the application .

**3) Use async messaging**

- The producer may not even know who is consuming the message .
- New services can easily consume the messages without any modifications to the producer.

**4) Don't build domain knowledge into a gateway**

Gateway is useful just for~~~~ ~~~~things like :

- request routing
- protocol translation
- load balancing
- authentication

should be restricted to this sort of infrastructure functionality.

**5) Expose open interfaces**

- Avoid creating custom translations layers that sit between services.
- OPenAPi ..

**6) Design and tests against service contracts**

When services expose well-defined APIS, you can develop and test against those APIS .
That way, you can develop and test an individual service without spinning up all of its dependent services. (Of course,
you would still perform integration and load testing against)

**7) Abstract infrastructure away from domain logic**

Don't let domain logic get mixed up with infrastructure related functionality . Such as **messaging** or **persistence**
.

**8) Offload cross-cutting concerns toa separate service**

For example, if several services need to authenticate requests, you could moved this 