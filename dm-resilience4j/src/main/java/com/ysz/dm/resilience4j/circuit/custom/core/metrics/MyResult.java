package com.ysz.dm.resilience4j.circuit.custom.core.metrics;


public enum MyResult {
  BELOW_THRESHOLDS,
  FAILURE_RATE_ABOVE_THRESHOLDS,
  SLOW_CALL_RATE_ABOVE_THRESHOLDS,
  ABOVE_THRESHOLDS,
  BELOW_MINIMUM_CALLS_THRESHOLD;


  /**
   * <pre>
   *   是否 有任何的异常行为
   *   卡请求 和 挂请求 是否超出了 阈值
   *
   * </pre>
   * @param result 结果
   * @return
   */
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
