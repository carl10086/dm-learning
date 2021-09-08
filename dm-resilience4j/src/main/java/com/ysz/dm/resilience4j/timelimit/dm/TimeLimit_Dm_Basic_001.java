package com.ysz.dm.resilience4j.timelimit.dm;

import io.github.resilience4j.timelimiter.TimeLimiter;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.LockSupport;

public class TimeLimit_Dm_Basic_001 {

  private long longTask() {
    final long l = System.nanoTime();
    LockSupport.parkNanos(2_000L);
    return System.nanoTime() - l;
  }

  public void execute() throws Exception {
    final ExecutorService executorService = Executors.newFixedThreadPool(2);
    final Future<Long> submit = executorService.submit(() -> longTask());
    TimeLimiter timeLimiter = TimeLimiter.of(Duration.ofSeconds(1));
    final Long aLong = timeLimiter.executeFutureSupplier(() -> submit);
    System.out.println(aLong);
    executorService.shutdown();
  }

  public static void main(String[] args) throws Exception{
    new TimeLimit_Dm_Basic_001().execute();
  }

}
