package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MySnapshot;
import java.util.concurrent.TimeUnit;

public interface MyMetrics {

  /**
   * <pre>
   *   records a call
   * </pre>
   *
   * @param duration the duration of the call
   * @param durationUnit the time unit of the duration
   * @param outcome the outcome of the call
   * @return
   */
  MySnapshot record(
      long duration,
      TimeUnit durationUnit,
      MyOutcome outcome
  );

  /**
   * <pre>
   *   returns a snapshot
   * </pre>
   * @return a snapshot
   */
  MySnapshot getSnapshot();

  enum MyOutcome {
    SUCCESS, ERROR, SLOW_SUCCESS, SLOW_ERROR
  }


}
