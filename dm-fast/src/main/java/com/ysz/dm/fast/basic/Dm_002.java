package com.ysz.dm.fast.basic;

import java.util.Arrays;

public class Dm_002 {

  private String s1;

  private void m2() {
    System.out.println(this.s1);

  }


  /**
   * <pre>
   *   可变参数:
   *   1. 本质就是个数组
   *   char...
   * </pre>
   */
  public static void m1(
      int i1, int i2, String... params
  ) {
    /* lambda */
    Arrays.stream(params).forEach(System.out::println);
  }

  public static void main(String[] args) throws Exception {
    new Dm_002().m2();

    /**/
    Dm_002.m1(1, 2, "a");
  }
}

