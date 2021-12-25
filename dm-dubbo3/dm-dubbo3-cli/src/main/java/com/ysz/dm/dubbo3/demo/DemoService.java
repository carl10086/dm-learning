package com.ysz.dm.dubbo3.demo;

import java.util.concurrent.CompletableFuture;

@javax.annotation.Generated(
    value = "by Dubbo generator",
    comments = "Source: demoservice.proto")
public interface DemoService {

  static final String JAVA_SERVICE_NAME = "com.ysz.dm.dubbo3.demo.DemoService";
  static final String SERVICE_NAME = "demoservice.DemoService";

  // FIXME, initialize Dubbo3 stub when interface loaded, thinking of new ways doing this.
  static final boolean inited = DemoServiceDubbo.init();

  HelloReply sayHello(HelloRequest request);

  CompletableFuture<HelloReply> sayHelloAsync(HelloRequest request);


}
