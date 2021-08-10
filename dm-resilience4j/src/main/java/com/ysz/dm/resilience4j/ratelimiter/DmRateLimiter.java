package com.ysz.dm.resilience4j.ratelimiter;

import static java.lang.System.nanoTime;
import static java.util.concurrent.locks.LockSupport.parkNanos;

import com.ysz.dm.resilience4j.ratelimiter.config.DmRateLimitCfg;
import com.ysz.dm.resilience4j.ratelimiter.core.DmRateLimiterState;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;

public class DmRateLimiter {

  private AtomicReference<DmRateLimiterState> state;

  private AtomicInteger waitingThreads = new AtomicInteger(0);


  private static final long nanoTimeStart = nanoTime();


  /**
   * @param permits 请求需要的 permits
   * @return
   */
  public boolean acPermission(int permits) {

    /*1. 最大超时时间.*/
    final long timeoutInNanos = state.get().getCfg().getTimeoutDuration().toNanos();

    /*2.cas 计算并且修改 state, 这里感觉 cas ? ... 可能好吧，可能恰到好处 ...*/
    final DmRateLimiterState modifiedState = casUpdateState(permits, timeoutInNanos);

    boolean result = waitForPermissionIfNecessary(
        timeoutInNanos,
        modifiedState.getNanosToWait()
    );

    return result;
  }

  private boolean waitForPermissionIfNecessary(
      final long timeoutInNanos,
      final long nanosToWait) {

    /*1. <=0 表示不需要等待*/
    boolean canAcquireImmediately = nanosToWait <= 0;

    if (canAcquireImmediately) {
      /*不需要等待*/
      return true;
    }

    /*2. 是否能到在 timeout 前成功达到*/
    boolean canAcquireInTime = timeoutInNanos >= nanosToWait;

    if (canAcquireInTime) {
      return waitForPermissions(nanosToWait);
    }

    /*3. 还是会等待这么多时间、而且 permit 不会过去 .*/
    waitForPermissions(timeoutInNanos);
    return false;
  }

  /**
   * 不中断的等待这么长时间
   * @param nanosToWait 需要等待的时间
   * @return true 表示中间没有被中断
   */
  private boolean waitForPermissions(final long nanosToWait) {
    /*1. 等待 thread cnt ++*/
    waitingThreads.incrementAndGet();

    /*2. 计算最后等待的 deadline*/
    long deadline = currentNanoTime() + nanosToWait;

    /*3. 这个变量用来记录、中间是否被打断、这里就算被打断还是会继续 睡眠下去*/
    boolean wasInterrupted = false;
    while (currentNanoTime() < deadline && !wasInterrupted) {
      long sleepBlockDuration = deadline - currentNanoTime();
      parkNanos(sleepBlockDuration);
      wasInterrupted = Thread.interrupted();
    }

    waitingThreads.decrementAndGet();
    if (wasInterrupted) {
      /*严谨、 恢复对应的信号 ....*/
      Thread.currentThread().interrupt();
    }

    return !wasInterrupted;
  }


  private DmRateLimiterState casUpdateState(
      int permits,
      long timeoutInNanos
  ) {
    DmRateLimiterState prev;
    DmRateLimiterState next;

    do {
      prev = state.get();
      next = calNextState(permits, timeoutInNanos, prev);

    } while (!compareAndSet(prev, next));

    return next;
  }

  private DmRateLimiterState calNextState(
      final int permits,
      final long timeoutInNanos,
      final DmRateLimiterState cur) {
    /*1. 获取 时间周期 as ns*/
    final long cyclePeriodInNanos = cur.getCfg().getLimitRefreshPeriod().toNanos();
    /*2. 获取 一个时间周期 中允许的 limit 数*/
    final int permissionsPerCycle = cur.getCfg().getLimitForPeriod();

    /*3. 获取当前时间(本质上是一个相对时间)*/
    long currentNanos = currentNanoTime();
    long currentCycle = currentNanos / cyclePeriodInNanos;

    /*4. 获取 上一次的 cycle 和 permits*/
    long nextCycle = cur.getActiveCycle();
    int nextPermits = cur.getActivePermissions();

    /*5. 开始计算不在一个周期的情况*/
    if (nextCycle != currentCycle) {
      /*5.1 如果上一个周期已经过了，那么上一次的这个就没啥意义了*/
      long elapsedCycles = currentCycle - nextCycle; /*must > 0, 需要假定时间不回滚*/

      /*5.2 这里累积计算了 中间 时间周期 累积的 permits*/
      long accumulatedPermissions = elapsedCycles * permissionsPerCycle;

      /*5.3 时间周期肯定要设置 为当前的值*/
      nextCycle = currentCycle;

      /*5.4 核心: 计算的是上一次. 个人感觉这里没有必要 ... 前面的肯定 > 后面的 ... */

      /*5.4.1 逻辑推理、如下:  nextCycle - currentCycle >= 1 可以推出 accumulatedPermissions>=permissionsPerCycle */
      nextPermits = (int) Long.min(nextPermits + accumulatedPermissions, permissionsPerCycle);
    } else {
      /*6. 计算 在同一个周期的情况 . nextPermits < permissionsPerCycle. 所以不用计算 */
      // nextPermits = (int) Long.min(nextPermits, permissionsPerCycle);
    }



    /*7. 计算需要等待的时间*/
    long nextNanosToWait = nanosToWaitForPermission(
        permits, cyclePeriodInNanos, permissionsPerCycle, nextPermits, currentNanos,
        currentCycle
    );

    /*8. 根据需要等待的时间计算 .*/
    DmRateLimiterState state = reservePermissions(
        cur.getCfg(),
        permits,
        timeoutInNanos,
        nextCycle,
        nextPermits,
        nextNanosToWait
    );

    return state;
  }


  private DmRateLimiterState reservePermissions(
      final DmRateLimitCfg config,
      final int permits,
      final long timeoutInNanos,
      final long cycle,
      final int availablePermissions /*当前剩余的 permissions*/,
      final long nanosToWait) {



    /*1. 判断 要等待的时间 是否 > 最大等待时间*/
    boolean canAcquireInTime = timeoutInNanos >= nanosToWait;

    int permissionsWithReservation = availablePermissions;

    /*2. 如果在这个时间内能达到目标、 就 - 法*/
    if (canAcquireInTime) {
      permissionsWithReservation -= permits;
    }

    DmRateLimiterState dmRateLimiterState = new DmRateLimiterState();
    dmRateLimiterState.setCfg(config);
    dmRateLimiterState.setActiveCycle(cycle);
    dmRateLimiterState.setActivePermissions(permissionsWithReservation);
    dmRateLimiterState.setNanosToWait(nanosToWait);
    return dmRateLimiterState;


  }


  /**
   * <pre>
   *  计算 申请 permit  个 需要等待多少时间.
   *
   *  思路: 分为2种情况.
   *
   *  1. 情况1: 当前还有剩余 > permits, 不用等待
   *  2. 情况2: 当前剩余的 < permits . 此时要等待的时间为:
   *             - 当前时间周期剩余的时间全部搞完
   *             - 等待 permit 需要的时间周期. (ceil 就行)
   *
   *
   * </pre>
   * @param permits 申请的 permit
   * @param cyclePeriodInNanos 时间周期
   * @param permissionsPerCycle 一个时间周期允许的 permit
   * @param availablePermissions 剩下的 permit
   * @param currentNanos 当前的时间
   * @param currentCycle 当前时间处于的 周期数
   * @return
   */
  private long nanosToWaitForPermission(
      final int permits,
      final long cyclePeriodInNanos,
      final int permissionsPerCycle,
      final int availablePermissions,
      final long currentNanos,
      final long currentCycle) {

    /*1. 如果 还有许可、不需要等待*/
    if (availablePermissions >= permits) {
      return 0L;
    }


    /*2. 计算下一次的时间周期 到点时间*/
    long nextCycleTimeInNanos = (currentCycle + 1) * cyclePeriodInNanos;
    /*3. 计算到下一个时间周期还要多久. 精确到 ns*/
    /*3.1 这个是个准确时间*/
    long nanosToNextCycle = nextCycleTimeInNanos - currentNanos;

    /*4. 在下一个时间周期开始还有几个 permits = 当前周期剩下的 + 一个周期的*/
    int permissionsAtTheStartOfNextCycle = availablePermissions + permissionsPerCycle;

    /*5. 计算需要等待多少个  cycle*/
    int fullCyclesToWait = divCeil(-(permissionsAtTheStartOfNextCycle - permits),
        permissionsPerCycle);

    /*6. */
    return (fullCyclesToWait * cyclePeriodInNanos) + nanosToNextCycle;
  }


  private boolean compareAndSet(final DmRateLimiterState cur, final DmRateLimiterState next) {

    if (state.compareAndSet(cur, next)) {
      return true;
    }
    /*就是为了 park 1 ns ..  */
    LockSupport.parkNanos(1);
    return false;
  }


  private long currentNanoTime() {
    return nanoTime() - nanoTimeStart;
  }


  private static int divCeil(int x, int y) {
    return (x + y - 1) / y;
  }


}
