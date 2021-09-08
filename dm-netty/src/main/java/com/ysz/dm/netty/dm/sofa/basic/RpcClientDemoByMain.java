package com.ysz.dm.netty.dm.sofa.basic;

import com.alipay.remoting.rpc.RpcClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RpcClientDemoByMain {

  static Logger logger = LoggerFactory
      .getLogger(RpcClientDemoByMain.class);


  static String addr = "127.0.0.1:8999";


  public static void main(String[] args) {
    RpcClient client = new RpcClient();
    client.startup();
    try {
      final Object carl = client.invokeSync(addr, "carl", 1000);
      System.err.println(carl);
    } catch (Exception e) {
      e.printStackTrace();
    }

    client.shutdown();
  }

}
