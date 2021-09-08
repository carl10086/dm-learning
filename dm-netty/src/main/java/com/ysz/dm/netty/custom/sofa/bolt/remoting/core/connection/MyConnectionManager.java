package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection;

import com.alipay.remoting.Connection;

public interface MyConnectionManager {

  Connection get(String poolKey);
}
