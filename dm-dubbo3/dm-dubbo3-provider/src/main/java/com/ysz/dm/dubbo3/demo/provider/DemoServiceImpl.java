package com.ysz.dm.dubbo3.demo.provider;

import com.ysz.dm.dubbo3.demo.DemoService;
import com.ysz.dm.dubbo3.demo.HelloReply;
import com.ysz.dm.dubbo3.demo.HelloRequest;
import java.util.concurrent.CompletableFuture;
import org.apache.dubbo.rpc.RpcContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DemoServiceImpl implements DemoService {

  private Logger logger = LoggerFactory.getLogger(DemoServiceImpl.class);

  @Override
  public HelloReply sayHello(HelloRequest request) {
    logger
        .info("Hello " + request.getName() + ", request from consumer: " + RpcContext.getContext().getRemoteAddress());
    return HelloReply.newBuilder()
        .setMessage("Hello " + request.getName() + ", response from provider: "
            + RpcContext.getContext().getLocalAddress())
        .build();
  }

  @Override
  public CompletableFuture<HelloReply> sayHelloAsync(HelloRequest request) {
    return CompletableFuture.completedFuture(sayHello(request));
  }
}
