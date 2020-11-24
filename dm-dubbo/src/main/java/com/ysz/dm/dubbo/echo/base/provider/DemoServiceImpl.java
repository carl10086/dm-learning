package com.ysz.dm.dubbo.echo.base.provider;

import com.ysz.dm.dubbo.echo.base.api.DemoService;
import java.util.concurrent.CompletableFuture;
import org.apache.dubbo.rpc.RpcContext;

public class DemoServiceImpl implements DemoService {

  @Override
  public String sayHello(String name) {
    return "Hello " + name + ", response from provider: " + RpcContext.getContext()
        .getLocalAddress();
  }

  @Override
  public CompletableFuture<String> sayHelloAsync(String name) {
    return null;
  }
}
