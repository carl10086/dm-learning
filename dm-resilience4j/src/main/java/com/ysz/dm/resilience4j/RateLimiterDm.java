package com.ysz.dm.resilience4j;

import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterConfig;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import java.text.SimpleDateFormat;
import java.time.Duration;
import java.util.Date;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class RateLimiterDm {


  private Semaphore semaphore = new Semaphore(10);

  private AtomicInteger counter = new AtomicInteger(0);

  private RateLimiter rateLimiter;


  // Create registry
  private RateLimiterRegistry rateLimiterRegistry = RateLimiterRegistry
      .of(RateLimiterConfig.custom()
          .limitRefreshPeriod(Duration.ofSeconds(1))
          .limitForPeriod(2)
          .timeoutDuration(Duration.ofDays(1))
          .build());

  public RateLimiterDm() {
    this.rateLimiter = rateLimiterRegistry.rateLimiter("rateLimit");
    int threadNum = 50;
    /*2. start */
    sleep(2000);
    for (int i = 0; i < threadNum; i++) {
      new Thread(new Worker(this.rateLimiter, this.counter)).start();
    }
  }

  public static void main(String[] args) throws Exception {
    new RateLimiterDm();
    System.in.read();
  }


  private static class Worker implements Runnable {


    private RateLimiter rateLimiter;
    private final AtomicInteger counter;

    public Worker(final RateLimiter rateLimiter,
        final AtomicInteger counter) {
      this.rateLimiter = rateLimiter;
      this.counter = counter;
    }

    @Override
    public void run() {
//      while (true) {
      try {
        rateLimiter.acquirePermission();
        addCounter();
//          sleep(1000);
      } catch (Exception e) {
        e.printStackTrace();
      } finally {
      }
    }
//    }


    private void addCounter() {
      System.out
          .printf("%s: this count:%s\n", new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()),
              this.counter.incrementAndGet());
    }


  }


  private static void sleep(long millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException ignored) {
    }
  }

}
