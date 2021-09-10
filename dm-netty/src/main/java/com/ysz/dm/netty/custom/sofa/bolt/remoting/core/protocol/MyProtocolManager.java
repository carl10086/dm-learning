package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol;

import java.util.HashMap;
import java.util.Map;

/**
 * <pre>
 *   维护所有的 protocols , code 和 protocol 的映射关系
 * </pre>
 */
public class MyProtocolManager {

  /**
   * <pre>
   *   感觉没有解决线程安全问题的必要，只要写入方保证单线程即可 .
   * </pre>
   */
  private static final Map<MyProtocolCode, MyProtocol> protocols = new HashMap<>();


  public static void registerProtocol(MyProtocol protocol, byte... protocolCodeBytes) {
    registerProtocol(protocol, MyProtocolCode.fromBytes(protocolCodeBytes));
  }

  public static void registerProtocol(MyProtocol protocol, MyProtocolCode protocolCode) {
    if (null == protocolCode || null == protocol) {
      throw new RuntimeException("Protocol: " + protocol + " and protocol code:"
          + protocolCode + " should not be null!");
    }
    MyProtocol exists = MyProtocolManager.protocols.putIfAbsent(protocolCode, protocol);
    if (exists != null) {
      throw new RuntimeException("Protocol for code: " + protocolCode + " already exists!");
    }
  }

  public static MyProtocol getProtocol(MyProtocolCode protocolCode) {
    return protocols.get(protocolCode);
  }

}
