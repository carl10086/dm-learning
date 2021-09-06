package com.ysz.dm.resilience4j.circuit.custom.core.exceptions;

import com.ysz.dm.resilience4j.circuit.custom.MyCircuitBreaker;

public class MyCallNotPermittedException extends RuntimeException {

  private static final long serialVersionUID = 3474455768050069960L;

  private final transient String causingCircuitBreakerName;


  private MyCallNotPermittedException(MyCircuitBreaker circuitBreaker, String message,
      boolean writableStackTrace) {
    super(message, null, false, writableStackTrace);
    this.causingCircuitBreakerName = circuitBreaker.getName();
  }

  /**
   * 工厂类
   * @param circuitBreaker 熔断器 ..
   * @return
   */
  public static MyCallNotPermittedException createCallNotPermittedException(
      MyCircuitBreaker circuitBreaker) {
    boolean writableStackTraceEnabled = circuitBreaker.getCircuitBreakerCfg()
        .isWritableStackTraceEnabled();

    String message = String
        .format("CircuitBreaker '%s' is %s and does not permit further calls",
            circuitBreaker.getName(), circuitBreaker.getState());

    return new MyCallNotPermittedException(circuitBreaker, message, writableStackTraceEnabled);
  }

}
