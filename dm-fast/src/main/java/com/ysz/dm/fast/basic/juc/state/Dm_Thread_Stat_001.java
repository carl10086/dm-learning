package com.ysz.dm.fast.basic.juc.state;

/**
 * @author carl
 */
public class Dm_Thread_Stat_001 {

  public static void main(String[] args) throws Exception {
    WaitingThread waitingThread = new WaitingThread();
    ThreadStatPrinter threadStatPrinter = new ThreadStatPrinter(waitingThread);
    threadStatPrinter.begin();
    waitingThread.start();
    Thread.sleep(5000L);
    waitingThread.wakeUp();
    threadStatPrinter.end();
  }


  private static class WaitingThread extends Thread {

    public void wakeUp() {
      synchronized (this) {
        this.notifyAll();
      }
    }

    @Override

    public void run() {
      synchronized (this) {
        try {
          System.out.printf("%s start wait\n", Thread.currentThread().getName());
          this.wait();
          System.out.printf("%s finish wait\n", Thread.currentThread().getName());
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
      }
    }
  }

}

