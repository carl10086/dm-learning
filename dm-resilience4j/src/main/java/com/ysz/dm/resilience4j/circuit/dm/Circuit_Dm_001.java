package com.ysz.dm.resilience4j.circuit.dm;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.SlidingWindowType;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.vavr.CheckedFunction0;
import io.vavr.control.Try;
import java.time.Duration;

public class Circuit_Dm_001 {

  CircuitBreakerConfig circuitBreakerConfig = CircuitBreakerConfig.custom()
      .failureRateThreshold(50)
      .slowCallRateThreshold(50)
      .waitDurationInOpenState(Duration.ofMillis(1000))
      .slowCallDurationThreshold(Duration.ofSeconds(2))
      .permittedNumberOfCallsInHalfOpenState(0)
      .minimumNumberOfCalls(10)
      .slidingWindowType(SlidingWindowType.COUNT_BASED)
      .slidingWindowSize(10)
//      .recordException(e -> INTERNAL_SERVER_ERROR
//          .equals(getResponse().getStatus()))
//      .recordExceptions(RuntimeException.class)
//      .ignoreExceptions(NullPointerException.class)
      .build();

  CircuitBreakerRegistry circuitBreakerRegistry = CircuitBreakerRegistry.of(circuitBreakerConfig);

  public void execute() {
    CircuitBreaker c1 = circuitBreakerRegistry.circuitBreaker("name1");

    for (int i = 0; i < 20; i++) {
      final int count = i;
      final CheckedFunction0<Integer> stringCheckedFunction0 = c1
          .decorateCheckedSupplier((CheckedFunction0<Integer>) () -> {
            int j = 0;
            return count / j;
          });

      try {
        Try.of(stringCheckedFunction0).get();
      } catch (Exception ignored) {
      }
    }
  }


  public static void main(String[] args) throws Exception {
    new Circuit_Dm_001().execute();
  }

}
