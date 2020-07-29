package com.ysz.dm.fast.basic.juc.state;

/**
 * @author carl
 */
public class ThreadStatPrinter extends Thread {

  private volatile boolean close = false;
  private final Thread monitorThread;

  public ThreadStatPrinter(Thread monitorThread) {
    this.monitorThread = monitorThread;
  }

  @Override
  public void run() {
    while (!close) {
      System.out.printf("%s,state: %s\n",
          monitorThread.getName(),
          monitorThread.getState()
      );
      try {
        Thread.sleep(1000L);
      } catch (InterruptedException ignored) {
      }
    }
    System.out.printf("stop watch...\n");
  }

  public void begin() {
    this.start();
  }


  public void end() {
    this.close = true;
  }

}
