package com.ysz.dm.lib.resilience4j;

import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadFullException;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.retry.Retry;
import io.vavr.control.Try;
import java.util.function.Supplier;
import lombok.extern.slf4j.Slf4j;

/**
 * <pre>
 *  resi
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/23
 **/
@Slf4j
public class QuickStart {

  public static void main(String[] args) throws Exception {
    var backendService = new BackendService();
    var serviceGroup = "backendService";

    var circuitBreaker = CircuitBreaker.ofDefaults(serviceGroup);

    /*3 retry attempts* between 500ms*/
    var retry = Retry.ofDefaults(serviceGroup);

    var bulkHead = Bulkhead.ofDefaults(serviceGroup);

    Supplier<String> supplier = () -> backendService.slow(1, true);

    /*wrap with the circuit breaker*/
    var circuitSupplier = CircuitBreaker.decorateSupplier(circuitBreaker, supplier);

    /*wrap with the retry*/
    var retrySupplier = Retry.decorateSupplier(retry, circuitSupplier);

    /*use Try to */
    var result = BackendService.handleResult(Try.ofSupplier(retrySupplier));
    log.info("result :{}", result);
  }


}
