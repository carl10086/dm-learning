package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

import java.util.concurrent.TimeUnit;

public class MyFixedSizeSlidingWindowMetrics implements MyMetrics{

  @Override
  public MySnapshot record(final long duration, final TimeUnit durationUnit,
      final MyOutcome outcome) {
    return null;
  }

  @Override
  public MySnapshot getSnapshot() {
    return null;
  }
}
