package com.ysz.dm.resilience4j;

import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadConfig;
import io.github.resilience4j.bulkhead.BulkheadRegistry;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.vavr.CheckedFunction0;
import io.vavr.control.Try;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.Test;

/**
 * @author carl.yu
 * @date 2020/3/12
 */
public class UsageDm {

  private int badCall() {
    return 1 / 0;
  }

  public int slowCall() {
    System.out.println("start");
    try {
      Thread.sleep(3 * 1000L);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    System.out.println("finish");
    return 0;
  }

  @Test
  public void tstBulkHead() throws Exception {

    final BulkheadRegistry registry = BulkheadRegistry.of(BulkheadConfig.custom().build());

    ExecutorService executorService = Executors.newCachedThreadPool();
    Runnable task = () -> {
      final BulkheadConfig config = BulkheadConfig.custom()
          .maxConcurrentCalls(1)
          .build();
      Bulkhead bulkhead = registry.bulkhead("slowCall", config);
      Try<Integer> integerTry = Try.of(
          Bulkhead.decorateCheckedSupplier(bulkhead, (CheckedFunction0<Integer>) () -> slowCall())
      );
      if (integerTry.isSuccess()) {

      } else if (integerTry.isFailure()) {
        System.out.println(integerTry.failed().get());
      } else {
        System.out.println("success");
      }
    };

    for (int i = 0; i < 1000; i++) {
      executorService.execute(task);
    }

    System.in.read();
    executorService.shutdown();
  }

  @Test
  public void tstFallBackWithException() {
    CircuitBreakerConfig circuitBreakerConfig = CircuitBreakerConfig.custom()
        .failureRateThreshold(50)
        .slowCallRateThreshold(50)
        .waitDurationInOpenState(Duration.ofMillis(1000))
        .slowCallDurationThreshold(Duration.ofSeconds(2))
        .permittedNumberOfCallsInHalfOpenState(3)
        .minimumNumberOfCalls(10)
        .slidingWindowSize(5)
        .build();

    CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(circuitBreakerConfig);

    CircuitBreaker badCall = registry.circuitBreaker("badCall");
    CheckedFunction0<Integer> supplier = CircuitBreaker
        .decorateCheckedSupplier(badCall, () -> badCall());

    for (int i = 0; i < 10; i++) {
      Try<Integer> test = Try.of(supplier);
      if (test.isSuccess()) {
      } else {
        System.out.println(test.failed().get());
      }

    }

  }
}
