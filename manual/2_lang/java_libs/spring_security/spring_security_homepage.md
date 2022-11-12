
#java #java_lib #spring

## 1. intro

### 1.1 refer

[homepage](https://docs.spring.io/spring-security/reference/index.html)
[baeldung-spring-security](https://www.baeldung.com/security-spring)



> [!NOTE] What is Authentication ?
> 
Authentication is **how we verify the identity** of who is trying to access a particular resource .






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


![]()
![](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/multi-securityfilterchain.png)

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


- [So many filters](https://docs.spring.io/spring-security/reference/servlet/architecture.html#servlet-security-filters)


1. If u want to debug src code . `FilterChainProxy` is great . 
2. `FilterChainProxy` is more powerful than `Filter` because of `RequestMatcher` interface, you can use everything of HttpServletRequest for routing jobs .





### 2.3 Authentication


#### 2.3.0 quick start by self



> [!NOTE] Tips
> after open debug in annotation `@EnableWebSecurity(debug = true)` . you can see follows



```kotlin
Security filter chain: [
  DisableEncodeUrlFilter
  WebAsyncManagerIntegrationFilter
  SecurityContextPersistenceFilter
  HeaderWriterFilter
  LogoutFilter
  RequestCacheAwareFilter
  SecurityContextHolderAwareRequestFilter
  AnonymousAuthenticationFilter
  SessionManagementFilter
  ExceptionTranslationFilter
  **FilterSecurityInterceptor**
]

```


**SO what happen if authentication is failed . **


when we got authenticated failed . the path is as following -> `DelegatingAuthenticationEntryPoint` -> `BasicAuthenticationEntryPoint`

```bash
02:00:31.239 [http-nio-8080-exec-5] DEBUG o.s.s.w.a.DelegatingAuthenticationEntryPoint - **No match found. Using default entry point** org.springframework.security.web.authentication.www.BasicAuthenticationEntryPoint@1d591b15
// ....
```


the default implementation is : 

```java
@Override

public void commence(HttpServletRequest request, HttpServletResponse response,

AuthenticationException authException) throws IOException {

response.addHeader("WWW-Authenticate", "Basic realm=\"" + this.realmName + "\"");

response.sendError(HttpStatus.UNAUTHORIZED.value(), HttpStatus.UNAUTHORIZED.getReasonPhrase());

}
```



so . go to forward to another request `/error` . this is go to **BasicErrorController**

```
02:00:40.119 [http-nio-8080-exec-5] DEBUG o.s.web.servlet.DispatcherServlet - "ERROR" dispatch for POST "/error", parameters={}
02:00:40.119 [http-nio-8080-exec-5] DEBUG o.s.w.s.m.m.a.RequestMappingHandlerMapping - Mapped to org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController#error(HttpServletRequest)
```




> [!NOTE] The default entryPoint is redirect to error page
> IF we are a simple rest api , may we should just failed with json api .




#### 2.3.1 arch

mechanisms :


- username-password: how to authentication with  a  username/password
- oAuth2.0 Login : login with a openId connect and non-standard OAuth 2.0
- SAML 2.0 
- CAS Support : Central Authentication Server .
- Remember Me : How to remember a user past session expiration
- JAAS 
- OpenId
- X509 .
- Pre-Authentication : Only use spring security for authorization . *with authentication by other mechanism*


#### Persistence 






