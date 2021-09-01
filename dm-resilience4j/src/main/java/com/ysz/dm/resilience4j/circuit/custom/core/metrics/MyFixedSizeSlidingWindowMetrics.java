package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

import java.util.concurrent.TimeUnit;

public class MyFixedSizeSlidingWindowMetrics implements MyMetrics {

  /**
   * <pre>
   *   滑动窗口大小
   * </pre>
   */
  private int windowSize;

  /**
   * <pre>
   *   聚合 统计信息
   * </pre>
   */
  private MyTotalAggregation totalAggregation;

  /**
   * 一个个测量窗口
   */
  private MyMeasurement[] measurements;

  private int headIndex;

  public MyFixedSizeSlidingWindowMetrics(final int windowSize) {
    this.windowSize = windowSize;
    this.measurements = new MyMeasurement[windowSize];
    this.headIndex = 0;

    for (int i = 0; i < this.windowSize; i++) {
      measurements[i] = new MyMeasurement();
    }

    this.totalAggregation = new MyTotalAggregation();
  }

  @Override
  public MySnapshot record(
      final long duration,
      final TimeUnit durationUnit,
      final MyOutcome outcome) {
    /**
     * 这里必须加锁. 每次计算都会 + 锁 ; FIXME: 可能是比较有问题的性能点
     */
    synchronized (this) {
      totalAggregation.record(duration, durationUnit, outcome);
      moveWindowByOne().record(duration, durationUnit, outcome);
      return null;
    }
  }

  private MyMeasurement moveWindowByOne() {
    moveHeadIndexByOne();
    /*目得是为了释放 部分资源吧.*/
    final MyMeasurement latestMeasurement = this.measurements[headIndex];
    totalAggregation.removeBucket(latestMeasurement);
    latestMeasurement.reset();
    return latestMeasurement;
  }

  /**
   * Moves the headIndex to the next bucket.
   */
  void moveHeadIndexByOne() {
    this.headIndex = (headIndex + 1) % windowSize;
  }

  @Override
  public MySnapshot getSnapshot() {
    return null;
  }
}
