package com.ysz.dm.resilience4j.circuit.custom;

import com.google.common.base.Preconditions;
import com.ysz.dm.resilience4j.circuit.custom.core.MyCircuitBreakerCfg;
import com.ysz.dm.resilience4j.circuit.custom.core.MyCircuitBreakerMetrics;
import com.ysz.dm.resilience4j.circuit.custom.core.metrics.MyResult;
import io.github.resilience4j.circuitbreaker.internal.SchedulerFactory;
import io.vavr.Tuple;
import io.vavr.Tuple2;
import java.time.Clock;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

public class MyCircuitBreaker {

  /**
   * <pre>
   *   方便调试的命名
   * </pre>
   */
  private String name;

  private MyCircuitBreakerCfg circuitBreakerCfg;

  private AtomicReference<MyCircuitBreakerState> stateReference;

  private SchedulerFactory schedulerFactory;

  private Clock clock;

  public MyCircuitBreaker(
      final MyCircuitBreakerCfg circuitBreakerCfg,
      final AtomicReference<MyCircuitBreakerState> stateReference) {
    this.circuitBreakerCfg = Objects
        .requireNonNull(circuitBreakerCfg, "circuitBreakerCfg can't be null");
    this.stateReference = stateReference;

  }

  public String getName() {
    return name;
  }

  private void transitionToHalfOpenState() {
//    stateTransition(
//        MyState.HALF_OPEN,
//    );
  }

  private void transitionToOpenState() {
    stateTransition(
        MyState.OPEN,
        currentState
            ->
            new MyOpenState(currentState.attempts() + 1, currentState.getMetrics()));
  }

  /**
   * <pre>
   *   状态机转换核心代码 .
   * </pre>
   * @param newState 要称为的状态 ...
   * @param newStateGenerator
   */
  private void stateTransition(
      MyState newState,
      UnaryOperator<MyCircuitBreakerState> newStateGenerator
  ) {
    /*原子性操作、修改 state,  同时 ?*/
    final MyCircuitBreakerState prevState
        = stateReference.getAndUpdate(
        /*UnaryOperator 是特殊的 function, 输出和输入都是同一种类型*/
        currentState -> {
          /*1. 这里好像不会真正的修改、其实是对 state 的校验~*/
          MyStateTransition.transitionBetween(
              getName(),
              currentState.getState(),
              newState
          );
          /*2. 这里才是返回的是真正修改之后的 state*/
          /*2.1 到这一步就不需要进行 状态机判断了 ....*/
          return newStateGenerator.apply(currentState);
        }
    );
  }


  public static enum MyStateTransition {
    CLOSED_TO_CLOSED(MyCircuitBreaker.MyState.CLOSED, MyCircuitBreaker.MyState.CLOSED),
    CLOSED_TO_OPEN(MyCircuitBreaker.MyState.CLOSED, MyCircuitBreaker.MyState.OPEN),
    CLOSED_TO_DISABLED(MyCircuitBreaker.MyState.CLOSED, MyCircuitBreaker.MyState.DISABLED),
    CLOSED_TO_METRICS_ONLY(MyCircuitBreaker.MyState.CLOSED, MyCircuitBreaker.MyState.METRICS_ONLY),
    CLOSED_TO_FORCED_OPEN(MyCircuitBreaker.MyState.CLOSED, MyCircuitBreaker.MyState.FORCED_OPEN),
    HALF_OPEN_TO_HALF_OPEN(MyCircuitBreaker.MyState.HALF_OPEN, MyCircuitBreaker.MyState.HALF_OPEN),
    HALF_OPEN_TO_CLOSED(MyCircuitBreaker.MyState.HALF_OPEN, MyCircuitBreaker.MyState.CLOSED),
    HALF_OPEN_TO_OPEN(MyCircuitBreaker.MyState.HALF_OPEN, MyCircuitBreaker.MyState.OPEN),
    HALF_OPEN_TO_DISABLED(MyCircuitBreaker.MyState.HALF_OPEN, MyCircuitBreaker.MyState.DISABLED),
    HALF_OPEN_TO_METRICS_ONLY(MyCircuitBreaker.MyState.HALF_OPEN,
        MyCircuitBreaker.MyState.METRICS_ONLY),
    HALF_OPEN_TO_FORCED_OPEN(MyCircuitBreaker.MyState.HALF_OPEN,
        MyCircuitBreaker.MyState.FORCED_OPEN),
    OPEN_TO_OPEN(MyCircuitBreaker.MyState.OPEN, MyCircuitBreaker.MyState.OPEN),
    OPEN_TO_CLOSED(MyCircuitBreaker.MyState.OPEN, MyCircuitBreaker.MyState.CLOSED),
    OPEN_TO_HALF_OPEN(MyCircuitBreaker.MyState.OPEN, MyCircuitBreaker.MyState.HALF_OPEN),
    OPEN_TO_DISABLED(MyCircuitBreaker.MyState.OPEN, MyCircuitBreaker.MyState.DISABLED),
    OPEN_TO_METRICS_ONLY(MyCircuitBreaker.MyState.OPEN, MyCircuitBreaker.MyState.METRICS_ONLY),
    OPEN_TO_FORCED_OPEN(MyCircuitBreaker.MyState.OPEN, MyCircuitBreaker.MyState.FORCED_OPEN),
    FORCED_OPEN_TO_FORCED_OPEN(MyCircuitBreaker.MyState.FORCED_OPEN,
        MyCircuitBreaker.MyState.FORCED_OPEN),
    FORCED_OPEN_TO_CLOSED(MyCircuitBreaker.MyState.FORCED_OPEN, MyCircuitBreaker.MyState.CLOSED),
    FORCED_OPEN_TO_OPEN(MyCircuitBreaker.MyState.FORCED_OPEN, MyCircuitBreaker.MyState.OPEN),
    FORCED_OPEN_TO_DISABLED(MyCircuitBreaker.MyState.FORCED_OPEN,
        MyCircuitBreaker.MyState.DISABLED),
    FORCED_OPEN_TO_METRICS_ONLY(MyCircuitBreaker.MyState.FORCED_OPEN,
        MyCircuitBreaker.MyState.METRICS_ONLY),
    FORCED_OPEN_TO_HALF_OPEN(MyCircuitBreaker.MyState.FORCED_OPEN,
        MyCircuitBreaker.MyState.HALF_OPEN),
    DISABLED_TO_DISABLED(MyCircuitBreaker.MyState.DISABLED, MyCircuitBreaker.MyState.DISABLED),
    DISABLED_TO_CLOSED(MyCircuitBreaker.MyState.DISABLED, MyCircuitBreaker.MyState.CLOSED),
    DISABLED_TO_OPEN(MyCircuitBreaker.MyState.DISABLED, MyCircuitBreaker.MyState.OPEN),
    DISABLED_TO_FORCED_OPEN(MyCircuitBreaker.MyState.DISABLED,
        MyCircuitBreaker.MyState.FORCED_OPEN),
    DISABLED_TO_HALF_OPEN(MyCircuitBreaker.MyState.DISABLED, MyCircuitBreaker.MyState.HALF_OPEN),
    DISABLED_TO_METRICS_ONLY(MyCircuitBreaker.MyState.DISABLED,
        MyCircuitBreaker.MyState.METRICS_ONLY),
    METRICS_ONLY_TO_METRICS_ONLY(MyCircuitBreaker.MyState.METRICS_ONLY,
        MyCircuitBreaker.MyState.METRICS_ONLY),
    METRICS_ONLY_TO_CLOSED(MyCircuitBreaker.MyState.METRICS_ONLY, MyCircuitBreaker.MyState.CLOSED),
    METRICS_ONLY_TO_FORCED_OPEN(MyCircuitBreaker.MyState.METRICS_ONLY,
        MyCircuitBreaker.MyState.FORCED_OPEN),
    METRICS_ONLY_TO_DISABLED(MyCircuitBreaker.MyState.METRICS_ONLY,
        MyCircuitBreaker.MyState.DISABLED);


    private static final java.util.Map<Tuple2<MyState, MyState>, MyCircuitBreaker.MyStateTransition> STATE_TRANSITION_MAP = Arrays
        .stream(MyStateTransition.values())
        .collect(Collectors.toMap(v -> Tuple.of(v.fromState, v.toState), Function
            .identity()));

    private final MyCircuitBreaker.MyState fromState;
    private final MyCircuitBreaker.MyState toState;


    MyStateTransition(
        MyCircuitBreaker.MyState fromState,
        MyCircuitBreaker.MyState toState
    ) {
      this.fromState = fromState;
      this.toState = toState;
    }


    public static MyCircuitBreaker.MyStateTransition transitionBetween(
        String name,
        MyCircuitBreaker.MyState fromState,
        MyCircuitBreaker.MyState toState
    ) {
      MyCircuitBreaker.MyStateTransition stateTransition = STATE_TRANSITION_MAP.get(Tuple.of(
          fromState, toState
      ));

      Preconditions.checkNotNull(
          stateTransition,
          "null transition found, name:%s, from:%s, to:%s",
          name,
          fromState,
          toState
      );

      return stateTransition;
    }
  }


  public static enum MyState {
    DISABLED(3, false),
    METRICS_ONLY(5, true),
    CLOSED(0, true),
    OPEN(1, true),
    FORCED_OPEN(4, false),
    HALF_OPEN(2, true);

    public final boolean allowPublish;
    private final int order;

    private MyState(int order, boolean allowPublish) {
      this.order = order;
      this.allowPublish = allowPublish;
    }

    public int getOrder() {
      return this.order;
    }
  }


  private interface MyCircuitBreakerState {

    void acquirePermission();

    void onError(long duration, TimeUnit durationUnit, Throwable throwable);

    void onSuccess(long duration, TimeUnit durationUnit);

    MyState getState();

    MyCircuitBreakerMetrics getMetrics();

    int attempts();
  }


  private class MyClosedState implements MyCircuitBreakerState {

    private MyCircuitBreakerMetrics circuitBreakerMetrics;
    private AtomicBoolean isClosed;

    @Override
    public void acquirePermission() {

    }

    @Override
    public void onError(
        final long duration,
        final TimeUnit durationUnit,
        final Throwable throwable) {
//      checkIfThresholdsExceeded(circuitBreakerMetrics.onError(duration, durationUnit));
    }

    @Override
    public void onSuccess(final long duration, final TimeUnit durationUnit) {
      checkIfThresholdsExceeded(circuitBreakerMetrics.onSuccess(duration, durationUnit));
    }

    @Override
    public MyState getState() {
      return null;
    }

    @Override
    public MyCircuitBreakerMetrics getMetrics() {
      return null;
    }

    @Override
    public int attempts() {
      return 0;
    }

    private void checkIfThresholdsExceeded(MyResult result) {
      /*1. 如果超出了限制，意味着状态要改变. 当前 close 状态要换了*/
      if (MyResult.hasExceededThresholds(result)) {
        /*说明 closed 的作用.*/
        if (isClosed.compareAndSet(true, false)) {
//          publishCircuitThresholdsExceededEvent(result, circuitBreakerMetrics);
          transitionToOpenState();
        }
      }
    }
  }


  private class MyOpenState implements MyCircuitBreakerState {

    /**
     * <pre>
     *   为什么 openState 是要一个 attempts 的 .
     *
     *   这需要去了解 作者的思路、
     *
     *   尝试是因为会失败。 什么场景下会失败:
     *   -  推理就是 CAS .
     * </pre>
     */
    private int attempts;
    private Instant retryAfterWaitDuration;
    private MyCircuitBreakerMetrics circuitBreakerMetrics;
    private AtomicBoolean isOpen;
    private ScheduledFuture<?> transitionToHalfOpenFuture;

    public MyOpenState(
        int attempts,
        MyCircuitBreakerMetrics circuitBreakerMetrics
    ) {
      this.attempts = attempts; /*重试次数, 意味着有指数重试机制*/
      final Long waitDurationInMillis = circuitBreakerCfg.getWaitIntervalFunctionInOpenState()
          .apply(attempts);

      this.retryAfterWaitDuration = clock.instant().plus(waitDurationInMillis, ChronoUnit.MILLIS);
      this.circuitBreakerMetrics = circuitBreakerMetrics;

      /*可以看出这个配置是 自动开启 切换半 Open 状态的配置. 默认是关闭的*/
      if (circuitBreakerCfg.isAutomaticTransitionFromOpenToHalfOpenEnabled()) {
        final ScheduledExecutorService scheduledExecutorService = schedulerFactory.getScheduler();

        /*定时任务机制是为了过一段时间自动延迟进入 半开状态*/
        transitionToHalfOpenFuture = scheduledExecutorService.schedule(
            this::toHalfOpenState,
            waitDurationInMillis,
            TimeUnit.MILLISECONDS
        );
      } else {
        transitionToHalfOpenFuture = null;
      }
      isOpen = new AtomicBoolean(true);
    }


    private synchronized void toHalfOpenState() {
      if (isOpen.compareAndSet(true, false)) {
        transitionToHalfOpenState();
      }
    }


    @Override
    public void acquirePermission() {

    }

    @Override
    public void onError(final long duration, final TimeUnit durationUnit,
        final Throwable throwable) {

    }

    @Override
    public void onSuccess(final long duration, final TimeUnit durationUnit) {
      circuitBreakerMetrics.onSuccess(duration, durationUnit);
    }

    @Override
    public int attempts() {
      return this.attempts;
    }

    @Override
    public MyState getState() {
      return null;
    }

    @Override
    public MyCircuitBreakerMetrics getMetrics() {
      return null;
    }
  }
}
