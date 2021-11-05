package com.ysz.dm.fast.allocation.gct;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * <pre>
 *   内存分配器
 * </pre>
 */
public class ModelAllocator implements Runnable {

  private volatile boolean shutdown = false;

  private double chanceOfLongLived = 0.02;

  private int multiplierForLongLived = 20;

  private int x = 1024;

  private int y = 1024;

  /**
   * 控制内存分配率
   */
  private int mbPerSec = 50;

  private int shortLivedMs = 100;

  /**
   * 8个线程分配内存
   */
  private int nThreads = 8;

  private Executor exec = Executors.newFixedThreadPool(nThreads);

  @Override
  public void run() {
    final int mainSleep = (int) (1000.0 / mbPerSec);

    while (!shutdown) {
      for (int i = 0; i < mbPerSec; i++) {
        /*一个对象，分配了内存 : x 个 y ..*/
        ModelObjectAllocation to = new ModelObjectAllocation(x, y, lifetime());
        exec.execute(to);

        try {
          Thread.sleep(mainSleep);
        } catch (Exception e) {
          shutdown = true;
        }
      }
    }
  }

  public void shutdown() {
    this.shutdown = true;
  }

  public static void main(String[] args) throws Exception {
    final ModelAllocator modelAllocator = new ModelAllocator();
    final Thread thread = new Thread(modelAllocator);
    thread.start();
    thread.join();
  }


  /**
   * 用来模拟弱分代假说
   *
   * 也有寿命比较长的概率
   * @return
   */
  private int lifetime() {
    if (Math.random() < chanceOfLongLived) {
      /*这里是小概率、会触发长生命周期*/
      return multiplierForLongLived * shortLivedMs; // 20 * 100 ms
    }

    return shortLivedMs; // 100 ms
  }


  public static class ModelObjectAllocation implements Runnable {

    private final int[][] allocated;
    private final int lifeTime;


    public ModelObjectAllocation(int x, int y, int liveFor) {
      this.allocated = new int[x][y];
      this.lifeTime = liveFor;
    }

    @Override
    public void run() {
      try {
        Thread.sleep(lifeTime);
//      System.err.println(System.currentTimeMillis() + ": " + allocated.length);
      } catch (InterruptedException ex) {
      }
    }
  }
}
