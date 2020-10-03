package com.ysz.dm.resilience4j;

import io.github.resilience4j.timelimiter.TimeLimiter;
import io.github.resilience4j.timelimiter.TimeLimiterConfig;
import io.github.resilience4j.timelimiter.TimeLimiterRegistry;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

public class TimeLimiter_Dm_001 {


  private static String say() {
    System.err.println("current running thread:" + Thread.currentThread());
    try {
      Thread.sleep(1000L);
    } catch (Exception ignored) {
    }
    return "Hello World";
  }


  public static void main(String[] args) throws Exception {
    final TimeLimiterConfig config = TimeLimiterConfig.custom()
        .cancelRunningFuture(true)
        .timeoutDuration(Duration.ofMillis(500))
        .build();

    /*默认配置*/
    final TimeLimiterRegistry registry = TimeLimiterRegistry.of(config);

    final TimeLimiter tst = registry.timeLimiter("tst");

    final CompletableFuture<String> completableFuture = CompletableFuture.supplyAsync(
        () -> say()
    );
    System.out.println(tst.executeFutureSupplier(() -> completableFuture));
  }
}
