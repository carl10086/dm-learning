
## refer

[homepage](https://docs.spring.io/spring-security/reference/index.html)



## 1. intro




## 2.  Servlet Applications

[where is official sample](https://github.com/spring-projects/spring-security-samples/tree/5.7.x/servlet/spring-boot/java/hello-security)



### 2.1 Getting started .


**Spring boot features**


- **Auto Enables spring security default configuration**, which creates a servlet `filter` as a bean named `springSecurityFilterChain` . This bean is used for all things .
	- protecting the application URLS , validating submitted username and passwords, redirecting to the login form and so on  .
- Create a `UserDetailService` bean with a username of `user` and randomly generated password that is logged to the console .
- Registers the `Filter` with a bean named `SpringSecurityFilterChain` with the Servlet container for every request .



these things does a lot work . a summary of the features follows :

- require a authenticated user for any interaction with the application .
- Generate a `default login form` for you .  
- protects the `password` storage with BCrypt .
- lets the user log out .
- `CSRF` prevention 
- `Session Fixation` protection 
- Security Header integration
	- `HTTP Strict transport security` for secure requests .
	- `X-Content-Type-Options` integration
	- Cache Control (can be overridden later by your application to allow caching of your static resource)
	- `X-XSS-Protection` integration
	- X-Frame-Options integration to help prevent *Clickjacking*






**and auto integration with the Servlet Api methods** :


```java
HttpServletRequest#getRemoteUser()
HttpServletRequest.html#getUserPrincipal()
HttpServletRequest.html#isUserInRole(java.lang.String)
HttpServletRequest.html#login(java.lang.String, java.lang.String)
HttpServletRequest.html#logout()
```


### 2.2 Architecture

**1. Servlet Api has a filter chain mechanism**

```kotlin
fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
    // do something before the rest of the application
    chain.doFilter(request, response) // invoke the rest of the application
    // do something after the rest of the application
}
```


spring use a `DelegatingFilterProxy` that allows bridges between the servlet container and spring `ApplicationContext` .

![delegating filter proxy](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/delegatingfilterproxy.png)


```kotlin
fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
	// Lazily get Filter that was registered as a Spring Bean
	// For the example in DelegatingFilterProxy 
delegate
 is an instance of Bean Filter0
	val delegate: Filter = getFilterBean(someBeanName)
	// delegate work to the Spring Bean
	delegate.doFilter(request, response)
}
```



