package com.ysz.dm.resilience4j.circuit.custom.core;

import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MyMetrics;
import java.time.Clock;
import java.util.concurrent.atomic.LongAdder;

public class MyCircuitBreakerMetrics {

  private final MyMetrics metrics;
  private final float failureRateThreshold;
  private final float slowCallRateThreshold;
  private final long slowCallDurationThresholdInNanos;
  private final LongAdder numberOfNotPermittedCalls;
  private int minimumNumberOfCalls;

  public MyCircuitBreakerMetrics(
      int slidingWindowSize,
      MySlidingWindowType slidingWindowType,
      MyCircuitBreakerCfg circuitBreakerCfg,
      Clock clock
  ) {
    if (slidingWindowType == MySlidingWindowType.COUNT_BASED) {

    }
  }

}
