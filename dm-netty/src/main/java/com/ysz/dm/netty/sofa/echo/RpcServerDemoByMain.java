package com.ysz.dm.netty.sofa.echo;

import com.alipay.remoting.BizContext;
import com.alipay.remoting.rpc.RpcServer;
import com.alipay.remoting.rpc.protocol.SyncUserProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RpcServerDemoByMain {

  static Logger logger = LoggerFactory
      .getLogger(RpcServerDemoByMain.class);

  int port = 1234;

  RpcServer rpcServer;

  public RpcServerDemoByMain() {
    this.rpcServer = new RpcServer(port);
    this.rpcServer.registerUserProcessor(new SyncUserProcessor<SimpleStringWrapper>() {


      @Override
      public Object handleRequest(final BizContext bizCtx, final SimpleStringWrapper request)
          throws Exception {
        System.err.println("接收到请求...");
        return "hello";
      }

      @Override
      public String interest() {
        return SimpleStringWrapper.class.getName();
      }

    });
    if (rpcServer.start()) {
      System.out.println("server start ok!");
    } else {
      System.out.println("server start failed!");
    }

  }

  public static void main(String[] args) {
    new RpcServerDemoByMain();
  }

}
