package com.ysz.dm.lib.resilience4j;

import io.github.resilience4j.bulkhead.BulkheadRegistry;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;

/**
 * resilience 4j registry
 *
 * @author carl
 * @create 2022-10-25 11:23 AM
 **/
public class Resilience4jRegistry {

  private CircuitBreakerRegistry circuitBreakerRegistry;
  private BulkheadRegistry bulkheadRegistry;

  private static class Singleton {

    private static Resilience4jRegistry instance = new Resilience4jRegistry();
  }

}
