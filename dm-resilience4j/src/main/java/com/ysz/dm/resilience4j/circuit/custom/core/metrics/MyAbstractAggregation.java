package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MyMetrics.MyOutcome;
import java.util.concurrent.TimeUnit;

public abstract class MyAbstractAggregation {

  /**
   * <pre>
   * 所有的 call 总耗时
   * </pre>
   */
  long totalDurationInMillis = 0;

  /**
   * <pre>
   *   所有的慢查询 耗时
   * </pre>
   */
  int numberOfSlowCalls = 0;
  /**
   * <pre>
   *   所有的慢而且失败的 calls, 有的请求、不仅仅失败、还卡
   * </pre>
   */
  int numberOfSlowFailedCalls = 0;
  /**
   * 失败的 calls
   */
  int numberOfFailedCalls = 0;
  /**
   * <pre>
   *  所有的  call 数目
   * </pre>
   */
  int numberOfCalls = 0;


  void record(
      long duration,
      TimeUnit durationUnit,
      MyOutcome outcome) {
    /*1. 当前的 call + 1*/
    this.numberOfCalls++;
    /*2. 总时间 +*/
    this.totalDurationInMillis += durationUnit.toMillis(duration);
    /*3. */
    switch (outcome) {
      case ERROR:
        /*有错误*/
        numberOfFailedCalls++;
        break;
      case SLOW_SUCCESS:
        /*慢但是成功*/
        break;
      case SLOW_ERROR:
        /*慢而且有错误*/
        numberOfSlowCalls++;
        numberOfFailedCalls++;
        numberOfSlowFailedCalls++;
        break;
    }
  }

}
