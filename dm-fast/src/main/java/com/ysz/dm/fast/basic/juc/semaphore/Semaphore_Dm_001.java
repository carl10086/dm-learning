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
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
    }).start();
    System.in.read();
  }

}
