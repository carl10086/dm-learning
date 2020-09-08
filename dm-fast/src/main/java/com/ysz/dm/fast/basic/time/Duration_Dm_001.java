package com.ysz.dm.fast.basic.time;

public class Duration_Dm_001 {

  public static void main(String[] args) throws InterruptedException {
    long n1 = System.currentTimeMillis();
    long n2 = System.nanoTime();
    Thread.sleep(1000L);
    System.out.println(System.currentTimeMillis() - n1);
    System.out.println((System.nanoTime() - n2) / 1000000L);
  }
}
