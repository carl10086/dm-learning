## refer

- [homepage](https://spring.io/guides/gs/rest-service-cors/)

You can enable cross-origin resource sharing from either in individual controllers or globally.
The following topics describe how to do so:

**Controller method CORS configuration**

So that the RESTFul web service will include CORS access control headers in its response, you have to add
a `@CrossOrigin` annotations to the handler method, as the following listing , as the following shows:

```java
    @CrossOrigin(origins = "http://localhost:8080")
@GetMapping("/greeting")
public Greeting greeting(@RequestParam(required = false, defaultValue = "World") String name){
    System.out.println("==== get greeting ====");
    return new Greeting(counter.incrementAndGet(),String.format(template,name));
```

The @CrossOrigin annotation enables cross-origin resource sharing only for this specific method.

By default, it allows `all origins`, `all headers`, and `the HTTP methods specified in the annotation`. Also a `maxAge`
of 30 minutes is used. You can customize this behavior by specifying the value of one of the following annotation
attributes .

- `origins`
- `methods`
- `allowedHeadres`
- `exposedHeaders`
- `allowCredentials`
- `maxAge`

...


Global CORS configuration . 
In addition 
