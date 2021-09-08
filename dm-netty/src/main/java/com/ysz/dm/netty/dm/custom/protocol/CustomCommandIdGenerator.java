package com.ysz.dm.netty.dm.custom.protocol;

import java.util.concurrent.atomic.AtomicInteger;

public class CustomCommandIdGenerator {

  private static final AtomicInteger id = new AtomicInteger(0);

  /**
   * generate the next id
   *
   * @return
   */
  public static int nextId() {
    return id.updateAndGet(i -> i == Integer.MAX_VALUE ? 0 : i + 1);
  }

}
