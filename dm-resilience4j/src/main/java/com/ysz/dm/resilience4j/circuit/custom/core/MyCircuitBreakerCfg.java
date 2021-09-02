package com.ysz.dm.resilience4j.circuit.custom.core;

import com.google.common.base.Preconditions;
import io.github.resilience4j.core.IntervalFunction;
import java.time.Duration;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class MyCircuitBreakerCfg {


  private float failureRateThreshold = 50;
  private int permittedNumberOfCallsInHalfOpenState = 10;
  private int slidingWindowSize = 100;
  private MySlidingWindowType slidingWindowType = MySlidingWindowType.COUNT_BASED;
  private int minimumNumberOfCalls = 100;
  private boolean writableStackTraceEnabled = true;
  private boolean automaticTransitionFromOpenToHalfOpenEnabled = false;
  /**
   * <pre>
   *   实际上是状态机转换时的函数. 输入
   * </pre>
   */
  private IntervalFunction waitIntervalFunctionInOpenState = IntervalFunction
      .of(Duration.ofSeconds(60));
  private float slowCallRateThreshold = 100;
  private Duration slowCallDurationThreshold = Duration
      .ofSeconds(60);


  private MyCircuitBreakerCfg() {
  }


  public static class Builder {


    private float failureRateThreshold;

    /**
     * <pre>
     *   半开状态的时候允许通过的 call num
     * </pre>
     */
    private int permittedNumberOfCallsInHalfOpenState = 10;

    /**
     * <pre>
     *   配置滑动窗口的大小.
     *   - 当状态机是关闭状态的时候、用来记录 call 统计结果
     * </pre>
     */
    private int slidingWindowSize = 100;

    /**
     * <pre>
     *   滑动窗口的纬度、 默认是基于 最后N次呼叫 的结果
     * </pre>
     */
    private MySlidingWindowType slidingWindowType = MySlidingWindowType.COUNT_BASED;

    /**
     * <pre>
     *   配置CircuitBreaker计算错误率或慢速呼叫率之前所需的最小呼叫数（每个滑动窗口期）。
     *
     *   eg: 如果minimumNumberOfCalls是10，那么在计算故障率之前，必须至少记录10次呼叫。
     *   - 如果只记录了9个呼叫，即使9个呼叫都失败了，断路器也不会过渡到开放。
     *
     * </pre>
     */
    private int minimumNumberOfCalls = 100;

    /**
     * <pre>
     *   暂时未开放配置
     * </pre>
     */
    private boolean writableStackTraceEnabled = true;

    private boolean automaticTransitionFromOpenToHalfOpenEnabled = false;

    /**
     * 默认的 等待 interval
     */
    private IntervalFunction waitIntervalFunctionInOpenState = IntervalFunction
        .of(Duration.ofSeconds(60));


    /**
     * <pre>
     *   配置一个百分比的阈值。当呼叫持续时间大于slowCallDurationThreshold时，CircuitBreaker认为该呼叫是慢速的。
     *   当慢速呼叫的百分比等于或大于阈值时，CircuitBreaker过渡到开放并开始短路呼叫。
     * </pre>
     */
    private float slowCallRateThreshold = 100;

    /**
     * <pre>
     *   默认 > 60s 的查询会被当做慢查询
     * </pre>
     */
    private Duration slowCallDurationThreshold = Duration
        .ofSeconds(60);

    /**
     * <pre>
     *   设置半开状态下允许通过的 call 数目
     * </pre>
     * @param permittedNumberOfCallsInHalfOpenState >=1
     * @return
     */
    public Builder permittedNumberOfCallsInHalfOpenState(
        final int permittedNumberOfCallsInHalfOpenState) {
      Preconditions.checkState(
          permittedNumberOfCallsInHalfOpenState > 0,
          "permittedNumberOfCallsInHalfOpenState must be greater than 0"
      );
      this.permittedNumberOfCallsInHalfOpenState = permittedNumberOfCallsInHalfOpenState;
      return this;
    }

    /**
     * <pre>
     *   设置滑动窗口类型
     *   这三个参数实际上要配套设置.
     *
     *   1. 如果是 counter based 的计算方式、那么 minimumNumberOfCalls 会取 minimumNumberOfCalls 和 slidingWindowSize 的最小值 . 也就是说 minimumNumberOfCalls <= slidingWindowSize 才有意义
     * </pre>
     * @param slidingWindowSize 断路器关闭时滑动窗口的大小
     * @param minimumNumberOfCalls 在计算故障率之前必须记录的最小呼叫数
     * @param slidingWindowType 滑动窗口的类型。要么是COUNT_BASED或者TIME_BASED。
     * @return
     */
    public Builder slidingWindow(
        final int slidingWindowSize,
        final int minimumNumberOfCalls,
        MySlidingWindowType slidingWindowType
    ) {
      Preconditions.checkArgument(
          slidingWindowSize > 0,
          "slidingWindowSize must be greater than 0"
      );

      Preconditions.checkArgument(
          minimumNumberOfCalls > 0,
          "minimumNumberOfCalls must be greater than 0"
      );

      Preconditions.checkArgument(
          slidingWindowType != null,
          "slidingWindowType can't be null"
      );

      if (slidingWindowType == MySlidingWindowType.COUNT_BASED) {
        /*注意: 在 counterBased 的时候, minimumNumberOfCalls 受到窗口的约束, 也就是说 minimumNumberOfCalls <= slidingWindowSize 才有意义. 不能 > 1个窗口*/
        this.minimumNumberOfCalls = Math.min(
            minimumNumberOfCalls,
            slidingWindowSize
        );
      } else {
        this.minimumNumberOfCalls = minimumNumberOfCalls;
      }

      this.slidingWindowSize = slidingWindowSize;
      this.slidingWindowType = slidingWindowType;
      return this;
    }

    public Builder failureRateThreshold(float failureRateThreshold) {
      if (failureRateThreshold <= 0 || failureRateThreshold > 100) {
        throw new IllegalArgumentException(
            "failureRateThreshold must be between 1 and 100");
      }
      this.failureRateThreshold = failureRateThreshold;
      return this;
    }


    public MyCircuitBreakerCfg build() {
      MyCircuitBreakerCfg config = new MyCircuitBreakerCfg();
      config.waitIntervalFunctionInOpenState = waitIntervalFunctionInOpenState;
      config.slidingWindowType = slidingWindowType;
      config.failureRateThreshold = failureRateThreshold;
      config.slowCallDurationThreshold = slowCallDurationThreshold;
      config.slowCallRateThreshold = slowCallRateThreshold;
      config.slidingWindowSize = slidingWindowSize;
      config.minimumNumberOfCalls = minimumNumberOfCalls;
      config.permittedNumberOfCallsInHalfOpenState = permittedNumberOfCallsInHalfOpenState;
      config.automaticTransitionFromOpenToHalfOpenEnabled = automaticTransitionFromOpenToHalfOpenEnabled;
      config.writableStackTraceEnabled = writableStackTraceEnabled;
      return config;
    }
  }

}
