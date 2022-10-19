## intro

async messaging is widely used, and provides many benefits, but also bring challenges such as the ordering of messages,
posion message management, idempotency, and mode .

Considerations :

mq tech likes `redis`, `rabbitmq`, `kafka` .

- subscription ;
- Security ;
- Subsets of the messages ;
- Topics ;
- Content filtering ;
- BI-directional communication: 双向通信 . The channels in a publish-subscribe system are treated as unidirectional. If
  a specific subscriber needs to send ack or communicate status back to the publisher, consider using the Req/Reply
  patterns ;
- Message ordering ;
- Message proiority ;
- Poison messages ;
- Repeated messages ;
- Message expiration ;
- Message scheduling ;