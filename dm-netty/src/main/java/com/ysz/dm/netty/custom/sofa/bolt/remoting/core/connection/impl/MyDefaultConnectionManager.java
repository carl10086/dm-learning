package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.impl;

import com.alipay.remoting.Connection;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.MyConnectionManager;

public class MyDefaultConnectionManager implements MyConnectionManager {

  @Override
  public Connection get(final String poolKey) {
    return null;
  }
}
