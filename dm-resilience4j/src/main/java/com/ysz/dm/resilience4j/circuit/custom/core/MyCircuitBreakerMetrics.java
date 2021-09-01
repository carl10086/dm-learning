package com.ysz.dm.resilience4j.circuit.custom.core;

import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MyMetrics;
import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MyMetrics.MyOutcome;
import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MyResult;
import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MySnapshot;
import java.time.Clock;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

public class MyCircuitBreakerMetrics {

  /**
   * <pre>
   *   2种滑动窗口解耦的地方、应该也是最核心的实现
   * </pre>
   */
  private MyMetrics metrics;
  private float failureRateThreshold;
  private float slowCallRateThreshold;
  /**
   * <pre>
   *   慢查询阈值, 单位纳秒
   * </pre>
   */
  private long slowCallDurationThresholdInNanos;
  private LongAdder numberOfNotPermittedCalls;
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


  public MyResult onSuccess(
      long duration,
      TimeUnit durationUnit
  ) {
    final MySnapshot snapshot;

    snapshot = metrics.record(duration, durationUnit,
        durationUnit.toNanos(duration) > slowCallDurationThresholdInNanos ?
            MyOutcome.SLOW_SUCCESS :
            MyOutcome.SUCCESS
    );

    return null;

  }

}
