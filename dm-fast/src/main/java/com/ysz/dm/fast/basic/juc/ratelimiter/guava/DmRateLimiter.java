package com.ysz.dm.fast.basic.juc.ratelimiter.guava;

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.math.LongMath;
import com.ysz.dm.fast.basic.juc.ratelimiter.guava.core.DmSleepingStopwatch;

public class DmRateLimiter {

  private final DmSleepingStopwatch stopwatch;
  private volatile Object mutexDoNotUseDirectly;


  /**
   * <pre>
   *   当前可用的 permits
   * </pre>
   */
  private double storedPermits;


  /**
   * <pre>
   *   开始前允许的最大 permit
   * </pre>
   */
  double maxPermits;

  /**
   * 在我们的稳定速率下，两个单位请求之间的时间间隔。例如，稳定速率为每秒5次的许可，其稳定间隔为200ms。
   */
  double stableIntervalMicros;


  /**
   * 下一个请求（无论其大小）将被批准的时间。在批准了一个请求后，这个时间会被进一步推到未来。大的请求比小的请求推得更远。
   */
  private long nextFreeTicketMicros = 0L; // could be either in the past or future

  private DmRateLimiter(DmSleepingStopwatch stopwatch) {
    this.stopwatch = stopwatch;
  }


  /**
   * 懒加载对象锁的技巧之一
   *  volatile + synchronized
   */
  private Object mutex() {
    Object mutex = mutexDoNotUseDirectly;
    if (mutex == null) {
      synchronized (this) {
        mutex = mutexDoNotUseDirectly;
        if (mutex == null) {
          mutexDoNotUseDirectly = mutex = new Object();
        }
      }
    }
    return mutex;
  }


  /**
   * <pre>
   *   从这个RateLimiter获取给定数量的许可，阻断直到请求可以被批准。如果有的话，说明睡眠时间的数量。
   * </pre>
   *
   * @param permits 申请的许可数目
   * @return 为执行速率所花费的睡眠时间，单位是秒；如果没有速率限制，则为0.0
   */
  public double acquire(int permits) {
    /*1. 计算需要 permits ， 最多需要等待多少时间*/
    long microsToWait = reserve(permits);
    /*2. 睡眠这么时间*/
    stopwatch.sleepMicrosUninterruptibly(microsToWait);
    /*3. 把等待的时间转化为秒返回*/
    return 1.0 * microsToWait / SECONDS.toMicros(1L);
  }


  /**
   * <pre>
   *   从这个RateLimiter中保留给定数量的许可，以备将来使用，并返回到保留量可以被消耗的微秒数。
   * </pre>
   * @param permits
   * @return
   */
  final long reserve(int permits) {
    checkPermits(permits);
    synchronized (mutex()) {
      return reserveAndGetWaitLength(permits, stopwatch.readMicros());
    }
  }


  final long reserveAndGetWaitLength(int permits, long nowMicros) {
    long momentAvailable = reserveEarliestAvailable(permits, nowMicros);
    return max(momentAvailable - nowMicros, 0);
  }

  void resync(long nowMicros) {
    /*1. 如果下一个请求被批准的时间在过去.*/
    if (nowMicros > nextFreeTicketMicros) {
      /*2. 这个时间间隔内能有多少新的 permit !*/
      double newPermits = (nowMicros - nextFreeTicketMicros) / coolDownIntervalMicros();
      /*3. 新的 permit + 上之后， 不能超过 maPermits*/
      storedPermits = min(maxPermits, storedPermits + newPermits);
      /*4. 调整为当前时间 .*/
      nextFreeTicketMicros = nowMicros;
    }
  }

  /**
   * <pre>
   *   保留所要求的许可证数量，并返回这些许可证可以使用的时间（有一个注意事项）。
   * </pre>
   * @param requiredPermits 需要的 permits
   * @param nowMicros 当前的时间
   * @return
   */
  private final long reserveEarliestAvailable(int requiredPermits, long nowMicros) {
    /*1. 重新同步 nextFreeTicketMicros,  此时 nextFreeTicketMicros >= nowMicros*/
    resync(nowMicros);
    long returnValue = nextFreeTicketMicros;
    /*2. 计算要使用多少 permit */
    double storedPermitsToSpend = min(requiredPermits, this.storedPermits);
    /*3. 计算 不够的话 要申请多少新的 permit*/
    double freshPermits = requiredPermits - storedPermitsToSpend;

    /*4. smooth 的话 就是 = 0  +  需要的 permit * 稳定的时间间隔(每个请求的周期单位) */
    long waitMicros = storedPermitsToWaitTime(this.storedPermits, storedPermitsToSpend)
        + (long) (freshPermits * stableIntervalMicros);

    /*5. 此时 nextFree = nextFreeTicketMicros + waitMicros, 意味着可以消耗未来的时间 */
    this.nextFreeTicketMicros = LongMath.saturatedAdd(nextFreeTicketMicros, waitMicros);

    /*6. 用了这么多许可 要减掉*/
    this.storedPermits -= storedPermitsToSpend;
    return returnValue;
  }

  private static void checkPermits(int permits) {
    checkArgument(permits > 0, "Requested permits (%s) must be positive", permits);
  }

  private double coolDownIntervalMicros() {
    return stableIntervalMicros;
  }

  private long storedPermitsToWaitTime(double storedPermits, double permitsToTake) {
    return 0L;
  }
}
