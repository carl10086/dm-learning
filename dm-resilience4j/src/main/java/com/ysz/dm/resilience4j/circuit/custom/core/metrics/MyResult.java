package com.ysz.dm.resilience4j.circuit.custom.core.metrics;


public enum MyResult {
  BELOW_THRESHOLDS,
  FAILURE_RATE_ABOVE_THRESHOLDS,
  SLOW_CALL_RATE_ABOVE_THRESHOLDS,
  ABOVE_THRESHOLDS,
  BELOW_MINIMUM_CALLS_THRESHOLD;


  public static boolean hasExceededThresholds(MyResult result) {
    return
        hasFailureRateExceededThreshold(result)
            ||
            hasSlowCallRateExceededThreshold(result);
  }

  public static boolean hasFailureRateExceededThreshold(MyResult result) {
    return result == ABOVE_THRESHOLDS || result == FAILURE_RATE_ABOVE_THRESHOLDS;
  }

  public static boolean hasSlowCallRateExceededThreshold(MyResult result) {
    return result == ABOVE_THRESHOLDS || result == SLOW_CALL_RATE_ABOVE_THRESHOLDS;
  }
}
