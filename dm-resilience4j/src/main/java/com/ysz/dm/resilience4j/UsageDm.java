package com.ysz.dm.resilience4j;

import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadConfig;
import io.github.resilience4j.bulkhead.BulkheadRegistry;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.timelimiter.TimeLimiter;
import io.github.resilience4j.timelimiter.TimeLimiterConfig;
import io.github.resilience4j.timelimiter.TimeLimiterRegistry;
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
    try {
      Thread.sleep(3 * 1000L);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return 0;
  }

  @Test
  public void tstBulkHead() throws Exception {

    final BulkheadRegistry registry = BulkheadRegistry.of(BulkheadConfig.custom().build());

    ExecutorService executorService = Executors.newCachedThreadPool();

    /*主程序调用了多个子线程， 可以理解为多个 HTTP IO 处理线程*/
    Runnable task = () -> {
      final BulkheadConfig config = BulkheadConfig.custom()
          .maxConcurrentCalls(3)
          .build();
      Bulkhead bulkhead = registry.bulkhead("slowCall", config);
      /*在子线程就迅速失败的东西. 这样调用者线程就不会阻塞住了 ..*/
      Try<Integer> integerTry = Try.of(
          Bulkhead.decorateCheckedSupplier(bulkhead, (CheckedFunction0<Integer>) () -> slowCall())
      );
      if (integerTry.isSuccess()) {
        System.out.printf("success get result:%s\n", integerTry.get());
      } else if (integerTry.isFailure()) {
        System.err.println(integerTry.failed().get());
      } else {
        System.out.println("success");
      }
    };

    for (int i = 0; i < 20; i++) {
      executorService.execute(task);
      Thread.sleep(10L);
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
        final Throwable throwable = test.failed().get();
        if (throwable instanceof CallNotPermittedException) {
          throwable.printStackTrace();
        } else {
          System.out.println(throwable);
        }
      }

    }

  }


  @Test
  public void tstTimeout() {

    final TimeLimiterRegistry registry = TimeLimiterRegistry
        .of(TimeLimiterConfig.custom().cancelRunningFuture(true)
            .timeoutDuration(Duration.ofMillis(500L)).build());

    final TimeLimiter timeLimiter = registry.timeLimiter("name");

//    timeLimiter.executeFutureSupplier();
  }
}
