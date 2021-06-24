package com.ysz.dm.web.metrics.example;

import com.ysz.dm.web.common.Utils;
import org.junit.Test;

public class ExampleGaugeReporterTest {

  @Test
  public void tstGet() {
    ExampleGaugeReporter reporter = new ExampleGaugeReporter();

    for (int i = 0; i < 1000; i++) {
      Utils.trySleep(100L);
      System.out.println(reporter.report());
    }
  }
}