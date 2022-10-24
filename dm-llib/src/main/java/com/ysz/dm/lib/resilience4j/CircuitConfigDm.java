package com.ysz.dm.lib.resilience4j;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.SlidingWindowType;
import io.vavr.control.Try;
import java.util.function.Supplier;
import lombok.extern.slf4j.Slf4j;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/23
 **/
@Slf4j
public class CircuitConfigDm {

  private CircuitBreakerConfig circuitBreakerConfig;
  private CircuitBreaker circuitBreaker;

  public CircuitConfigDm() {
    this.circuitBreakerConfig = CircuitBreakerConfig
        .custom()
//        .slidingWindow(50, 50, SlidingWindowType.COUNT_BASED)
        .build();
    this.circuitBreaker = CircuitBreaker.of("custom", circuitBreakerConfig);
  }

  private String execute() {
    Supplier<String> supplier = () -> BackendService.slowWithMills(10, true);
    var circuitBreaker = CircuitBreaker.decorateSupplier(this.circuitBreaker, supplier);
    return BackendService.handleResult(Try.ofSupplier(circuitBreaker));
  }


  public static void main(String[] args) throws Exception {
    CircuitConfigDm circuitConfigDm = new CircuitConfigDm();
    for (int i = 0; i < 200; i++) {
      log.info("times:{}, result:{}", i, circuitConfigDm.execute());
    }
  }


}
