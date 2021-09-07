package com.ysz.dm.resilience4j.circuit.custom.core;

import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MyFixedSizeSlidingWindowMetrics;
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


  /**
   * <pre>
   *   create from Half Open ...
   *   1. 此时窗口转化为 COUNT_BASED
   *   2. 窗口大小上限调整为 permittedNumberOfCallsInHalfOpenState, 但是不可能超过 cfg 的 getMinimumNumberOfCalls
   * </pre>
   * @param permittedNumberOfCallsInHalfOpenState openState 中允许的请求
   * @param circuitBreakerCfg cfg
   * @param clock clock
   * @return
   */
  public static MyCircuitBreakerMetrics forHalfOpen(
      final int permittedNumberOfCallsInHalfOpenState,
      final MyCircuitBreakerCfg circuitBreakerCfg,
      final Clock clock
  ) {
    return new MyCircuitBreakerMetrics(
        permittedNumberOfCallsInHalfOpenState,
        MySlidingWindowType.COUNT_BASED,
        circuitBreakerCfg,
        clock
    );
  }

  /**
   *
   * @param slidingWindowSize 滑动窗口大小
   * @param slidingWindowType 滑动窗口类型
   * @param circuitBreakerCfg 配置
   * @param clock
   */
  public MyCircuitBreakerMetrics(
      int slidingWindowSize,
      MySlidingWindowType slidingWindowType,
      MyCircuitBreakerCfg circuitBreakerCfg,
      Clock clock
  ) {
    /*1. 如果是基于计数器的 滑动窗口*/
    if (slidingWindowType == MySlidingWindowType.COUNT_BASED) {
      this.metrics = new MyFixedSizeSlidingWindowMetrics(slidingWindowSize);
      this.minimumNumberOfCalls = Math
          .min(slidingWindowSize, circuitBreakerCfg.getMinimumNumberOfCalls());
    }

    this.failureRateThreshold = circuitBreakerCfg.getFailureRateThreshold();
    this.slowCallRateThreshold = circuitBreakerCfg.getSlowCallRateThreshold();
    this.slowCallDurationThresholdInNanos = circuitBreakerCfg.getSlowCallDurationThreshold()
        .toNanos();
    this.numberOfNotPermittedCalls = new LongAdder();
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

    return checkIfThresholdsExceeded(snapshot);
  }

  public MyResult onError(
      long duration,
      TimeUnit durationUnit
  ) {
    return checkIfThresholdsExceeded(metrics.record(
        duration,
        durationUnit,
        durationUnit.toNanos(duration) > slowCallDurationThresholdInNanos ?
            MyOutcome.SLOW_ERROR :
            MyOutcome.ERROR
    ));

  }

  private MyResult checkIfThresholdsExceeded(
      MySnapshot snapshot
  ) {
    /*1. 获取目前的失败率，看上去 float 非常 ok*/
    float failureRateInPercentage = getFailureRate(snapshot);
    float slowCallsInPercentage = getSlowCallRate(snapshot);

    /*2. 这直接说明了上面特殊值 -1 的作用 是说明 目前的 call 还不够, 也就是处理窗口累积的逻辑在这里*/
    if (failureRateInPercentage == -1 || slowCallsInPercentage == -1) {
      return MyResult.BELOW_MINIMUM_CALLS_THRESHOLD;
    }

    if (failureRateInPercentage >= failureRateThreshold /*失败的占比要大*/
        && slowCallsInPercentage >= slowCallRateThreshold /*还同时慢的占比大, 也就是卡*/
    ) {
      /*3. 说明卡而且失败*/
      return MyResult.ABOVE_THRESHOLDS;
    }

    /*4.这个说明 仅仅挂的多*/
    if (failureRateInPercentage >= failureRateThreshold) {
      return MyResult.FAILURE_RATE_ABOVE_THRESHOLDS;
    }

    /*5.这个说明 仅仅卡的多*/
    if (slowCallsInPercentage >= slowCallRateThreshold) {
      return MyResult.SLOW_CALL_RATE_ABOVE_THRESHOLDS;
    }

    /*6. 正常，没问题*/
    return MyResult.BELOW_THRESHOLDS;
  }


  private float getSlowCallRate(final MySnapshot snapshot) {
    /*1. 和 getFailureRate 逻辑一样*/
    int bufferedCalls = snapshot.getTotalNumberOfCalls();
    if (bufferedCalls == 0 || bufferedCalls < minimumNumberOfCalls) {
      return -1.0f;
    }
    return snapshot.getSlowCallRate();
  }

  private float getFailureRate(final MySnapshot snapshot) {
    /*1. 总数*/
    int bufferedCalls = snapshot.getTotalNumberOfCalls();
    /*2. 如果总数 达不到最低标准、返回 -1 这个特殊值*/
    if (bufferedCalls == 0 || bufferedCalls < minimumNumberOfCalls) {
      return -1.0f;
    }

    return snapshot.getFailureRate();
  }

  public void onCallNotPermitted() {
    this.numberOfNotPermittedCalls.increment();
  }
}
