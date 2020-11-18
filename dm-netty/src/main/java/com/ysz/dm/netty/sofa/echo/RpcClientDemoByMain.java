package com.ysz.dm.netty.sofa.echo;

import com.alipay.remoting.InvokeContext;
import com.alipay.remoting.exception.RemotingException;
import com.alipay.remoting.rpc.RpcClient;

public class RpcClientDemoByMain {

  public static void main(String[] args) throws RemotingException, InterruptedException {
    RpcClient rpcClient = new RpcClient();
    rpcClient.init();
    final String addr = "127.0.0.1:1234";

    InvokeContext context = new InvokeContext();
    final Object hello = rpcClient
        .invokeSync(addr, new SimpleStringWrapper("hello"), context, 3000);
    System.err.println(hello);
  }
}
