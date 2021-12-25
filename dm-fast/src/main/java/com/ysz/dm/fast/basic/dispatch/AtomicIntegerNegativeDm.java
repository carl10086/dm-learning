package com.ysz.dm.fast.basic.dispatch;

import java.util.concurrent.atomic.AtomicInteger;

public class AtomicIntegerNegativeDm {


  public static void main(String[] args) {
    AtomicInteger counter = new AtomicInteger(Integer.MAX_VALUE - 50);
    for (int k = 0; k < 100; k++) {
      System.err.println(counter.updateAndGet(i -> i == Integer.MAX_VALUE ? 0 : i + 1));
    }

  }


}
