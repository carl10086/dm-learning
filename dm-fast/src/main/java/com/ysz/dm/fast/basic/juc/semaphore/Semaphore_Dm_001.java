package com.ysz.dm.fast.basic.juc.semaphore;

import java.util.concurrent.Semaphore;

public class Semaphore_Dm_001 {


  public static void main(String[] args) throws Exception {
    final Semaphore semaphore = new Semaphore(1);
    semaphore.acquire();
    Thread.sleep(10L);
    new Thread(() -> {
      try {
        semaphore.acquire();
//        final boolean b = semaphore.tryAcquire();
//        System.err.println(b);
        System.err.println("child thread can execute");
      } catch (Exception e) {
        e.printStackTrace();
      }
    }).start();

    System.out.println("parent thread begin to release");
    semaphore.release();
    System.out.println("parent thread finish to release");
    System.in.read();
  }

}
