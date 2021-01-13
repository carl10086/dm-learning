package com.ysz.dm.fast.jvm;

import java.util.ArrayList;
import java.util.List;

public class AllocateHelper {

  private static final int ONE_M = 1024 * 1024;

  private static List<byte[]> container = new ArrayList<>(10000);

  public static void allo(boolean remain) {
    byte[] bytes = new byte[1 * ONE_M];
    if (remain) {
      container.add(bytes);
    } else {
      bytes = null;
    }
  }
}
