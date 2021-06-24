package com.ysz.dm.web.metrics.example;

import java.util.concurrent.atomic.AtomicInteger;

public class ExampleCounterReporter {

  private AtomicInteger counter = new AtomicInteger();

  public void addAndGet() {
    counter.incrementAndGet();
  }
}
