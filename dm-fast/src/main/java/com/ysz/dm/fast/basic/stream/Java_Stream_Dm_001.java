package com.ysz.dm.fast.basic.stream;

import com.google.common.collect.Lists;

public class Java_Stream_Dm_001 {


  private static boolean filterTst(int x, int max) {
    return x <= max;
  }

  public static void main(String[] args) {
    final int max = 10;
    Lists.newArrayList(1, 2, 3, 4).stream().filter(x -> filterTst(x, max));
  }

}
