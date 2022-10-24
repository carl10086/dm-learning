package com.ysz.dm.lib.resilience4j;

import io.github.resilience4j.bulkhead.BulkheadFullException;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.vavr.control.Try;
import java.time.Duration;
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
public class BackendService {

  public static String alwaysFail() {
    return slow(0, true);
  }

  public static String slow(int seconds) {
    return slow(seconds, false);
  }

  public static String slowWithMills(int millis, boolean throwException) {
    if (millis > 0) {
      try {
        Thread.sleep(millis);
      } catch (Throwable ignored) {
      }
    }

    if (throwException) {
      throw new IllegalStateException("illegal state exception");
    }
    return "success call with " + millis;

  }

  public static String slow(int seconds, boolean throwException) {
    return slowWithMills((int) Duration.ofSeconds(seconds).toMillis(), throwException);
  }

  public static String handleResult(Try<String> result) {
    if (result.isSuccess()) {
      /*result is success*/
      return result.get();
    } else if (result.failed().get() instanceof CallNotPermittedException) {
      /*circuit breaker*/
      log.error("circuit breaker happens");
      return null;
    } else if (result.failed().get() instanceof BulkheadFullException) {
      /*bulk head */
      log.error("bulk head  happens", result.getCause());
      return null;
    } else {
      Throwable cause = result.getCause();
      log.error("unknown exception, {}", cause.getClass());
      return null;
    }
  }
}
