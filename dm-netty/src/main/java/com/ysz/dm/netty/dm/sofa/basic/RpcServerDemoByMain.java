package com.ysz.dm.netty.dm.sofa.basic;

import com.alipay.remoting.BizContext;
import com.alipay.remoting.ConnectionEventType;
import com.alipay.remoting.rpc.RpcServer;
import com.alipay.remoting.rpc.protocol.SyncUserProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RpcServerDemoByMain {

  static Logger logger = LoggerFactory
      .getLogger(RpcServerDemoByMain.class);


  RpcServer server;
  int port = 8999;

  public RpcServerDemoByMain() {
    // 1. create a Rpc server with port assigned
    server = new RpcServer(this.port);

    // 2. add processor for connect and close event if you need
    server.addConnectionEventProcessor(ConnectionEventType.CONNECT,
        (remoteAddr, connection) -> logger.info("connection event, remoteAddr:{}", remoteAddr));
    server.addConnectionEventProcessor(ConnectionEventType.CLOSE,
        (remoteAddr, connection) -> logger.info("close event, remoteAddr:{}", remoteAddr));

    // 3. register user processor for client request
    server.registerUserProcessor(new SyncUserProcessor<String>() {
      @Override
      public Object handleRequest(final BizContext bizCtx, final String request) throws Exception {
        logger.warn("Request received:" + request + ", timeout:" + bizCtx.getClientTimeout()
            + ", arriveTimestamp:" + bizCtx.getArriveTimestamp());
        return "serverResp:" + request;
      }

      @Override
      public String interest() {
        return String.class.getName();
      }
    });
    // 4. server start
    server.startup();
    if (server.isStarted()) {
      System.out.println("server start ok!");
    } else {
      System.out.println("server start failed!");
    }
  }

  public static void main(String[] args) throws Exception {
    new RpcServerDemoByMain();
  }
}
