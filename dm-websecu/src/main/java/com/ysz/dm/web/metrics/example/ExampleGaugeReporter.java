package com.ysz.dm.web.metrics.example;

import com.ysz.dm.web.common.Utils;
import java.util.concurrent.atomic.AtomicInteger;

public class ExampleGaugeReporter {

  private AtomicInteger counter = new AtomicInteger(0);

  private Thread thread;

  private volatile boolean close = false;

  public ExampleGaugeReporter() {
    this.thread = new Thread(() -> {
      while (!close) {
        Utils.trySleep(10L);
        counter.addAndGet(1);
      }
    });
    this.thread.setDaemon(true);
    this.thread.start();
  }

  public double report() {
    return counter.get();
  }

  public void close() {
    this.close = true;
  }
}
