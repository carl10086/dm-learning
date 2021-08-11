package com.ysz.dm.fast.basic.juc.ratelimiter;

import com.codahale.metrics.ConsoleReporter;
import com.codahale.metrics.Meter;
import com.codahale.metrics.MetricRegistry;
import com.google.common.util.concurrent.RateLimiter;
import com.ysz.dm.fast.infra.tools.MixUtils;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;


/**
 * <pre>信号量的思路</pre>
 */
public class RateLimiterDm_Semap_002 {

  private final MetricRegistry metrics;
  private final Meter requests;

  private Semaphore semaphore = new Semaphore(10);

  private RateLimiter rateLimiter = RateLimiter.create(5);

  public RateLimiterDm_Semap_002() {
    metrics = new MetricRegistry();
    requests = metrics.meter("requests");

    /*1. 构造 metrics 工具*/
    ConsoleReporter reporter = ConsoleReporter.forRegistry(metrics)
        .convertRatesTo(TimeUnit.SECONDS)
        .convertDurationsTo(TimeUnit.MILLISECONDS)
        .build();
    reporter.start(1, TimeUnit.SECONDS);

    int threadNum = 50;
    /*2. start `*/
    for (int i = 0; i < threadNum; i++) {
      new Thread(new Worker(this.requests, this.semaphore, this.rateLimiter)).start();
    }

  }

  public static void main(String[] args) throws Exception {
    new RateLimiterDm_Semap_002();
    System.in.read();
  }


  private static class Worker implements Runnable {

    private final Meter requests;
    private Semaphore semaphore;
    private RateLimiter rateLimiter;

    private Worker(Meter requests, Semaphore semaphore,
        RateLimiter rateLimiter) {
      this.requests = requests;
      this.semaphore = semaphore;
      this.rateLimiter = rateLimiter;
    }

    @Override
    public void run() {
      while (true) {
        try {
          rateLimiter.acquire();
          MixUtils.sleep(1000);
          requests.mark();
        } catch (Exception e) {
          e.printStackTrace();
        } finally {
        }
      }
    }

  }

}
