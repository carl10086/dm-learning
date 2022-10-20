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

When a client consumes a service, the responsibility for enforing the business rules of the domain should not fall on
the client.

Otherwise, every client has to enforce the business rules , and you end up with domain knowledge spread across different
parts of the application .



**3) Use async messaging**


**4) Don't build domain knowledge into a gateway**


**5) Expose open interfaces**

- 
