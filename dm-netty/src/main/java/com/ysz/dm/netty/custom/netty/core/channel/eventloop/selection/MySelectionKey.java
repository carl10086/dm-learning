package com.ysz.dm.netty.custom.netty.core.channel.eventloop.selection;

import java.nio.channels.SelectableChannel;
import java.nio.channels.Selector;

public abstract class MySelectionKey {

  protected MySelectionKey() {
  }

  /**
   * <pre>
   *   返回创建 key 的 channel
   *   如果 key 已经 cancelled 了， 这个方法还是能返回对应的 channel
   * </pre>
   * @return
   */
  public abstract SelectableChannel channel();


  /**
   * <pre>
   *   返回创建 key 的 selector.
   *   同上 cancelled 了还是会返回
   * </pre>
   * @return
   */
  public abstract Selector selector();


  /**
   * <pre>
   *   测试对应的 key 是否是 valid 的 .
   * </pre>
   * @return
   */
  public abstract boolean isValid();


  public abstract void cancel();
}
