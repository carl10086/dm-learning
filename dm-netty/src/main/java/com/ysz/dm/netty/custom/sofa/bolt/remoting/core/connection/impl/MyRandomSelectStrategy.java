package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.impl;

import com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg.MyBoltOptionsCfg;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.MyConnection;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection.MyConnectionSelectStrategy;
import java.util.List;

public class MyRandomSelectStrategy implements MyConnectionSelectStrategy {


  private final MyBoltOptionsCfg options;

  public MyRandomSelectStrategy(MyBoltOptionsCfg options) {
    this.options = options;
  }

  @Override
  public MyConnection select(final List<MyConnection> connections) {
    return null;
  }
}
