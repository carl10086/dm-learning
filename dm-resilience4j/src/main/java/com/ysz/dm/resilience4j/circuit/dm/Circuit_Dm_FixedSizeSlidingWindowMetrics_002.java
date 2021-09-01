package com.ysz.dm.resilience4j.circuit.dm;

import io.github.resilience4j.core.metrics.FixedSizeSlidingWindowMetrics;
import io.github.resilience4j.core.metrics.Metrics.Outcome;
import java.util.concurrent.TimeUnit;

public class Circuit_Dm_FixedSizeSlidingWindowMetrics_002 {

  public void execute() {
    FixedSizeSlidingWindowMetrics metrics = new FixedSizeSlidingWindowMetrics(2);
    /*第一次成功请求是: 100ms */
    metrics.record(100L, TimeUnit.MILLISECONDS, Outcome.SUCCESS);
    /*第二次请求失败 : 100ms*/
    metrics.record(100L, TimeUnit.MILLISECONDS, Outcome.ERROR);
    /*第三次请求: 1000ms, slow_success*/
    metrics.record(1000L, TimeUnit.MILLISECONDS, Outcome.SLOW_SUCCESS);

  }

  public static void main(String[] args) throws Exception {
    new Circuit_Dm_FixedSizeSlidingWindowMetrics_002().execute();
  }
}
