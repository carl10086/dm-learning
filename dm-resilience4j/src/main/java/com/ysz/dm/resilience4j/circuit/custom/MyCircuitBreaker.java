package com.ysz.dm.resilience4j.circuit.custom;

import com.ysz.dm.resilience4j.circuit.custom.core.MyCircuitBreakerCfg;
import com.ysz.dm.resilience4j.circuit.custom.core.MyCircuitBreakerMetrics;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public class MyCircuitBreaker {

  private final MyCircuitBreakerCfg circuitBreakerCfg;

  private final AtomicReference<MyCircuitBreakerState> stateReference;

  public MyCircuitBreaker(
      final MyCircuitBreakerCfg circuitBreakerCfg,
      final AtomicReference<MyCircuitBreakerState> stateReference) {
    this.circuitBreakerCfg = Objects
        .requireNonNull(circuitBreakerCfg, "circuitBreakerCfg can't be null");
    this.stateReference = stateReference;

  }

  private interface MyCircuitBreakerState {

    void acquirePermission();

    void onError(long duration, TimeUnit durationUnit, Throwable throwable);

    void onSuccess(long duration, TimeUnit durationUnit);
  }


  private class MyClosedState implements MyCircuitBreakerState {

    private final MyCircuitBreakerMetrics circuitBreakerMetrics;
    private final AtomicBoolean isClosed;

    @Override
    public void acquirePermission() {

    }

    @Override
    public void onError(final long duration, final TimeUnit durationUnit,
        final Throwable throwable) {

    }

    @Override
    public void onSuccess(final long duration, final TimeUnit durationUnit) {

    }
  }

}
