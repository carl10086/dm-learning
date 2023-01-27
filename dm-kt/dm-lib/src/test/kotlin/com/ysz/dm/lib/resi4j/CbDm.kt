package com.ysz.dm.lib.resi4j

import io.github.resilience4j.circuitbreaker.CircuitBreaker
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry
import org.junit.jupiter.api.Test
import java.io.IOException
import java.time.Duration
import java.util.concurrent.TimeoutException


/**
 * @author carl
 * @create 2022-12-05 2:49 PM
 **/
internal class CbDm {

    private fun doSomething(): Unit {

    }

    @Test
    internal fun test_cb() {
        val circuitBreakerConfig = CircuitBreakerConfig.custom()
            .failureRateThreshold(50f)
            .waitDurationInOpenState(Duration.ofMillis(1000))
            .permittedNumberOfCallsInHalfOpenState(2)
            .slidingWindowSize(2)
            .recordExceptions(IOException::class.java, TimeoutException::class.java)
//            .ignoreExceptions(BusinessException::class.java, OtherBusinessException::class.java)
            .build()


        val circuitBreakerRegistry = CircuitBreakerRegistry.of(circuitBreakerConfig)

        val circuitBreaker: CircuitBreaker = circuitBreakerRegistry.circuitBreaker("name")




//        val decoratedSupplier: Supplier<String> = CircuitBreaker
//            .decorateSupplier<Any>(circuitBreaker, this::doSomething)
//
//
//        val result: String = Try.ofSupplier(decoratedSupplier)
//            .recover { throwable -> "Hello from Recovery" }.get()


    }
}