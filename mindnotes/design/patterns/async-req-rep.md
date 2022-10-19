Decouple backend processing from a frontend host, where backend processing needs to be async, but the frontend still
needs a clear response.

**Context And Problem**

In most cases, APIs of a client application are designed to respond quickly, on the order of 100 ms or less. Many
factors can affect the resp latency:

- An application's hosting stack .
- Security components
- The relative geographic location of the caller and the backend
- Network infra
- current load
- the size of the request payload
- Processing the queue length
- The time for the backend to process the request

Application code can make a sync API call in a non-blocking way, giving the appearance of async processing, which is
recommended for IO-bound operations .

**Solutions**

one solution to this problem is use HTTP polling. Polling is useful to client-side code, as it can be hard to provide
call-back endpoints or use long running connections .