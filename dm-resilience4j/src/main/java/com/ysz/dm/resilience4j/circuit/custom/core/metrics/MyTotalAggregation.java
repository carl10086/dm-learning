package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

public class MyTotalAggregation extends MyAbstractAggregation {

  public void removeBucket(final MyAbstractAggregation bucket) {
    this.totalDurationInMillis -= bucket.totalDurationInMillis;
    this.numberOfSlowCalls -= bucket.numberOfSlowCalls;
    this.numberOfSlowFailedCalls -= bucket.numberOfSlowFailedCalls;
    this.numberOfFailedCalls -= bucket.numberOfFailedCalls;
    this.numberOfCalls -= bucket.numberOfCalls;
  }
}
