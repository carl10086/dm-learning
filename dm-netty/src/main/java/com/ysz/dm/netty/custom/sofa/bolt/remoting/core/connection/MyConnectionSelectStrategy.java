package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.connection;

import java.util.List;

public interface MyConnectionSelectStrategy {

  MyConnection select(List<MyConnection> connections);

}
