package com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg;


public class MyBoltServerOptions {

  public static final MyBoltOption<Integer> TCP_SO_BACKLOG = MyBoltOption.valueOf(
      "bolt.tcp.so.backlog",
      1024);

  public static final MyBoltOption<Boolean> NETTY_EPOLL_LT = MyBoltOption.valueOf(
      "bolt.netty.epoll.lt",
      true);

  public static final MyBoltOption<Integer> TCP_SERVER_IDLE = MyBoltOption.valueOf(
      "bolt.tcp.server.idle.interval",
      90 * 1000);

  public static final MyBoltOption<Boolean> SERVER_MANAGE_CONNECTION_SWITCH = MyBoltOption.valueOf(
      "bolt.server.manage.connection",
      false);

  public static final MyBoltOption<Boolean> SERVER_SYNC_STOP = MyBoltOption.valueOf(
      "bolt.server.sync.stop",
      false);


}
