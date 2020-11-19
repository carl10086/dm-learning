package com.ysz.dm.netty.sofa.echo;

import com.alipay.remoting.InvokeContext;
import com.alipay.remoting.exception.RemotingException;
import com.alipay.remoting.rpc.RpcClient;
import com.alipay.remoting.rpc.RpcResponseFuture;

public class RpcClientDemoByMain {

  public static void main(String[] args) throws RemotingException, InterruptedException {
    RpcClient rpcClient = new RpcClient();
    rpcClient.init();
    final String addr = "127.0.0.1:1234";

    InvokeContext context = new InvokeContext();
    final SimpleStringWrapper request = new SimpleStringWrapper("hello");
    final int timeoutAsMs = 3000;
    System.err.println(rpcClient.invokeSync(addr, request, context, timeoutAsMs));
    final RpcResponseFuture responseFuture = rpcClient
        .invokeWithFuture(addr, request, context, timeoutAsMs);
    System.err.println(responseFuture.get());
  }
}
