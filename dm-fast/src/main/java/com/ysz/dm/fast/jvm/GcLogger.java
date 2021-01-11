package com.ysz.dm.fast.jvm;

import java.util.ArrayList;
import java.util.List;

public class GcLogger {

  private static final int ONE_M = 1024 * 1024;

  private static List<byte[]> container = new ArrayList<>(10000);

  private static void allo(boolean remain) {
    byte[] bytes = new byte[10 * ONE_M];
    if (remain) {
      container.add(bytes);
    } else {
      bytes = null;
    }
  }


  public static void main(String[] args) throws Exception {
    /*分配 100M*/
    for (int i = 0; i < 100; i++) {
      allo(i % 3 == 0);
    }
    System.in.read();
    System.out.println(container.size());
  }

}
