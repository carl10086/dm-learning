package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.command;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.MyInvokeContext;

/**
 * <pre>
 *   本质上应该叫做  MyAbstractRemotingCommand .
 *   所有请求响应对象共同的父类
 * </pre>
 */
public abstract class MyRpcCommand implements MyRemotingCommand {

  protected MyCommandCode cmdCode;
  protected byte version = 0x1;
  protected byte type;
  /**
   * <pre>
   *   使用的序列化工具
   * </pre>
   */
  protected byte serializer;
  protected int id;
  protected short clazzLength = 0;
  protected short headerLength = 0;
  protected int contentLength = 0;
  protected byte[] clazz;
  protected byte[] header;
  protected byte[] content;
  protected MyInvokeContext invokeContext;

}
