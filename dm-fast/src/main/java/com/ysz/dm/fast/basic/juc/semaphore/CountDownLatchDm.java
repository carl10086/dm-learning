package com.ysz.dm.fast.basic.juc.semaphore;

import java.util.concurrent.CountDownLatch;

public class CountDownLatchDm {

  private CountDownLatch countDownLatch = new CountDownLatch(1);

  public void exec() {
    final Thread thread = new Thread(new Inner(this));
    thread.start();
    try {
      Thread.sleep(1000L);
    } catch (InterruptedException ignored) {
    }

    this.countDownLatch.countDown();

  }

  private static class Inner implements Runnable {

    private final CountDownLatchDm countDownLatchDm;

    private Inner(final CountDownLatchDm countDownLatchDm) {
      this.countDownLatchDm = countDownLatchDm;
    }

    @Override
    public void run() {
      countDownLatchDm.tryWait();
    }
  }

  private void tryWait() {
    try {
      countDownLatch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) throws Exception {
    new CountDownLatchDm().exec();
    System.in.read();
  }

}
