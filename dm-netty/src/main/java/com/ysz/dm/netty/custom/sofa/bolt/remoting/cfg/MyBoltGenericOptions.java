package com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg;

import com.alipay.remoting.ConnectionSelectStrategy;
import com.ysz.dm.netty.custom.sofa.bolt.remoting.core.MyExtendedNettyChannelHandler;

/**
 * 服务端和客户端共享的配置项
 */
public class MyBoltGenericOptions {

  /**
   * <pre>
   *   默认开启 nodelay, 毕竟不想要延迟. 关闭 nagle 算法
   * </pre>
   */
  public static final MyBoltOption<Boolean> TCP_NODELAY = MyBoltOption.valueOf(
      "bolt.tcp.nodelay",
      true);

  /**
   * <pre>
   *   默认开启 reuseaddr ,更合理的释放端口资源
   * </pre>
   */
  public static final MyBoltOption<Boolean> TCP_SO_REUSEADDR = MyBoltOption.valueOf(
      "bolt.tcp.so.reuseaddr",
      true);

  /**
   * <pre>
   *   默认开启 tcp 本身的 keepAlive 选项
   * </pre>
   */
  public static final MyBoltOption<Boolean> TCP_SO_KEEPALIVE = MyBoltOption.valueOf(
      "bolt.tcp.so.keepalive",
      true);

  /**
   * <pre>
   *   默认关闭 sndbuf: 合理, 最好就别开放
   * </pre>
   */
  public static final MyBoltOption<Integer> TCP_SO_SNDBUF = MyBoltOption
      .valueOf("bolt.tcp.so.sndbuf");

  /**
   * <pre>
   *   默认关闭 rcvbuf: 合理 同上
   * </pre>
   */
  public static final MyBoltOption<Integer> TCP_SO_RCVBUF = MyBoltOption
      .valueOf("bolt.tcp.so.rcvbuf");

  /**
   * <pre>
   *
   * </pre>
   */
  public static final MyBoltOption<Integer> NETTY_IO_RATIO = MyBoltOption.valueOf(
      "bolt.netty.io.ratio",
      70);


  public static final MyBoltOption<Boolean> NETTY_BUFFER_POOLED = MyBoltOption.valueOf(
      "bolt.netty.buffer.pooled",
      true);

  /**
   * <pre>
   *   默认的写入高水位, 64K
   * </pre>
   */
  public static final MyBoltOption<Integer> NETTY_BUFFER_HIGH_WATER_MARK = MyBoltOption.valueOf(
      "bolt.netty.buffer.high.watermark",
      64 * 1024);

  /**
   * <pre>
   *   默认的写入低水位, 32K
   * </pre>
   */
  public static final MyBoltOption<Integer> NETTY_BUFFER_LOW_WATER_MARK = MyBoltOption.valueOf(
      "bolt.netty.buffer.low.watermark",
      32 * 1024);

  /**
   * <pre>
   *   暂时不考虑
   * </pre>
   */
  public static final MyBoltOption<Boolean> NETTY_EPOLL_SWITCH = MyBoltOption.valueOf(
      "bolt.netty.epoll.switch",
      true);

  public static final MyBoltOption<Boolean> TCP_IDLE_SWITCH = MyBoltOption.valueOf(
      "bolt.tcp.heartbeat.switch",
      true);
  /*------------ NETTY Config End ------------*/

  /*------------ Thread Pool Config Start ------------*/


  public static final MyBoltOption<Integer> TP_MIN_SIZE = MyBoltOption.valueOf(
      "bolt.tp.min",
      20);
  public static final MyBoltOption<Integer> TP_MAX_SIZE = MyBoltOption.valueOf(
      "bolt.tp.max",
      400);
  public static final MyBoltOption<Integer> TP_QUEUE_SIZE = MyBoltOption.valueOf(
      "bolt.tp.queue",
      600);
  public static final MyBoltOption<Integer> TP_KEEPALIVE_TIME = MyBoltOption.valueOf(
      "bolt.tp.keepalive",
      60);

  /*------------ Thread Pool Config End ------------*/

  public static final MyBoltOption<ConnectionSelectStrategy> CONNECTION_SELECT_STRATEGY = MyBoltOption
      .valueOf(
          "CONNECTION_SELECT_STRATEGY");

  public static final MyBoltOption<Boolean> NETTY_FLUSH_CONSOLIDATION = MyBoltOption.valueOf(
      "bolt.netty.flush_consolidation",
      false);

  public static final MyBoltOption<MyExtendedNettyChannelHandler> EXTENDED_NETTY_CHANNEL_HANDLER = MyBoltOption
      .valueOf(
          "bolt.extend.netty.channel.handler",
          null);


}
