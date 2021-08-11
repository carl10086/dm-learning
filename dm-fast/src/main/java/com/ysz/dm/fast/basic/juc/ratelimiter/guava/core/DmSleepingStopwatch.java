package com.ysz.dm.fast.basic.juc.ratelimiter.guava.core;

import static java.util.concurrent.TimeUnit.MICROSECONDS;

import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.Uninterruptibles;

public abstract class DmSleepingStopwatch {

  public DmSleepingStopwatch() {
  }

  /*
   * We always hold the mutex when calling this. TODO(cpovirk): Is that important? Perhaps we need
   * to guarantee that each call to reserveEarliestAvailable, etc. sees a value >= the previous?
   * Also, is it OK that we don't hold the mutex when sleeping?
   */
  public abstract long readMicros();

  public abstract void sleepMicrosUninterruptibly(long micros);


  /**
   * <pre>
   *   基于系统时间的 stopwatch
   * </pre>
   * @return
   */
  public static final DmSleepingStopwatch createFromSystemTimer() {
    return new DmSleepingStopwatch() {
      final Stopwatch stopwatch = Stopwatch.createStarted();

      @Override
      public long readMicros() {
        return stopwatch.elapsed(MICROSECONDS);
      }

      @Override
      public void sleepMicrosUninterruptibly(long micros) {
        if (micros > 0) {
          Uninterruptibles.sleepUninterruptibly(micros, MICROSECONDS);
        }
      }
    };
  }
}
