package com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg;


/**
 * <pre>
 *   封装所有的客户端配置常量
 * </pre>
 */
public class MyBoltClientOptions {

  public static final MyBoltOption<Integer> NETTY_IO_RATIO = MyBoltOption.valueOf(
      "bolt.tcp.heartbeat.interval",
      15 * 1000);
  public static final MyBoltOption<Integer> TCP_IDLE_MAXTIMES = MyBoltOption.valueOf(
      "bolt.tcp.heartbeat.maxtimes",
      3);

  public static final MyBoltOption<Integer> CONN_CREATE_TP_MIN_SIZE = MyBoltOption.valueOf(
      "bolt.conn.create.tp.min",
      3);
  public static final MyBoltOption<Integer> CONN_CREATE_TP_MAX_SIZE = MyBoltOption.valueOf(
      "bolt.conn.create.tp.max",
      8);
  public static final MyBoltOption<Integer> CONN_CREATE_TP_QUEUE_SIZE = MyBoltOption.valueOf(
      "bolt.conn.create.tp.queue",
      50);
  public static final MyBoltOption<Integer> CONN_CREATE_TP_KEEPALIVE_TIME = MyBoltOption.valueOf(
      "bolt.conn.create.tp.keepalive",
      60);

  public static final MyBoltOption<Boolean> CONN_RECONNECT_SWITCH = MyBoltOption.valueOf(
      "bolt.conn.reconnect",
      false);
  public static final MyBoltOption<Boolean> CONN_MONITOR_SWITCH = MyBoltOption.valueOf(
      "bolt.conn.monitor",
      false);

}
