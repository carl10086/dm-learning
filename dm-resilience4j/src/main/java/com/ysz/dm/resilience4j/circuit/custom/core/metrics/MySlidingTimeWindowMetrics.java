package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

import io.github.resilience4j.core.metrics.SnapshotImpl;
import java.time.Clock;
import java.util.concurrent.TimeUnit;

public class MySlidingTimeWindowMetrics implements MyMetrics {

  /**
   * 滑动窗口数组, 一秒对应一个窗口
   */
  MyPartialAggregation[] partialAggregations;

  /**
   * <pre>
   *  滑动窗口时间大小, 以 秒为单位.
   *    也就是说， 最小1秒
   * </pre>
   *
   */
  int timeWindowSizeInSeconds;

  /**
   * <pre>
   *   滑动窗口的总计数器
   * </pre>
   */
  MyTotalAggregation totalAggregation;

  /**
   * <pre>
   *   时间工具封装接口
   * </pre>
   */
  Clock clock;

  /**
   * <pre>
   *   当前窗口的指针
   * </pre>
   */
  int headIndex;

  public MySlidingTimeWindowMetrics(
      final int timeWindowSizeInSeconds,
      final Clock clock) {
    this.timeWindowSizeInSeconds = timeWindowSizeInSeconds;
    this.clock = clock;

    this.totalAggregation = new MyTotalAggregation();
    this.partialAggregations = new MyPartialAggregation[timeWindowSizeInSeconds];
    this.headIndex = 0;

    long epochSecond = clock.instant().getEpochSecond();
    for (int i = 0; i < timeWindowSizeInSeconds; i++) {
      partialAggregations[i] = new MyPartialAggregation(epochSecond);
      epochSecond++;
    }
  }

  @Override
  public MySnapshot record(
      final long duration,
      final TimeUnit durationUnit,
      final MyOutcome outcome) {
    synchronized (this) {
      totalAggregation.record(duration, durationUnit, outcome);
      moveWindowToCurrentEpochSecond(getLatestPartialAggregation())
          .record(duration, durationUnit, outcome);
      return new MySnapshotImpl(totalAggregation);
    }
  }

  private MyPartialAggregation moveWindowToCurrentEpochSecond(
      MyPartialAggregation latestPartialAggregation
  ) {
    final long currentEpochSecond = clock.instant().getEpochSecond();
    final long differenceInSeconds = currentEpochSecond - latestPartialAggregation.getEpochSecond();

    if (differenceInSeconds == 0) { //(1) 不用滑动, 还是当前 s
      return latestPartialAggregation;
    }

    long secondsToMoveTheWindow = Math.min(differenceInSeconds, timeWindowSizeInSeconds); //(2)

    MyPartialAggregation currentPartialAggregation;
    do {
      secondsToMoveTheWindow--; // (3)
      moveHeadIndexByOne(); // (4) 移动指针一个
      currentPartialAggregation = getLatestPartialAggregation(); //(5) 获取当前的 bucket
      totalAggregation.removeBucket(currentPartialAggregation); // (6) 总计数器移除 bucket
      currentPartialAggregation
          .reset(currentEpochSecond - secondsToMoveTheWindow); //(7) reset bucket
    } while (secondsToMoveTheWindow > 0);
    return currentPartialAggregation;
  }


  void moveHeadIndexByOne() {
    this.headIndex = (headIndex + 1) % timeWindowSizeInSeconds;
  }

  private MyPartialAggregation getLatestPartialAggregation() {
    return partialAggregations[headIndex];
  }


  @Override
  public MySnapshot getSnapshot() {
    return null;
  }
}
