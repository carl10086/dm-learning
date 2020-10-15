package com.ysz.dm.fast.basic.disruptor;

public class FalseSharing implements Runnable {

  public final static long ITERATIONS = 500L * 1000L * 100L;
  private int arrayIndex = 0;

  /**
   * 1. 公共临界资源、其长度是线程的长度
   *
   * 2. 所有的线程都修改自己对应 idx 的值
   */
  private static ValueNoPadding[] longs;

  public FalseSharing(final int arrayIndex) {
    this.arrayIndex = arrayIndex;
  }

  public static void main(final String[] args) throws Exception {
    for (int i = 1; i < 10; i++) {
      System.gc();
      final long start = System.currentTimeMillis();
      runTest(i);
      System.out.println("Thread num " + i + " duration = " + (System.currentTimeMillis() - start));
    }

  }

  private static void runTest(int NUM_THREADS) throws InterruptedException {
    Thread[] threads = new Thread[NUM_THREADS];
    longs = new ValueNoPadding[NUM_THREADS];
    for (int i = 0; i < longs.length; i++) {
      longs[i] = new ValueNoPadding();
    }
    for (int i = 0; i < threads.length; i++) {
      threads[i] = new Thread(new FalseSharing(i));
    }

    for (Thread t : threads) {
      t.start();
    }

    for (Thread t : threads) {
      t.join();
    }
  }

  public void run() {
    long i = ITERATIONS + 1;
    while (0 != --i) {
      longs[arrayIndex].value = 0L;
    }
  }

  public final static class ValuePadding {

    protected long p1, p2, p3, p4, p5, p6, p7;
    protected volatile long value = 0L;
    protected long p9, p10, p11, p12, p13, p14;
    protected long p15;
  }

  public final static class ValueNoPadding {

    // protected long p1, p2, p3, p4, p5, p6, p7;
    protected volatile long value = 0L;
    // protected long p9, p10, p11, p12, p13, p14, p15;
  }
}
