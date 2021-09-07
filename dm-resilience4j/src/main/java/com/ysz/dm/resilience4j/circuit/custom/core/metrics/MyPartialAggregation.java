package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class MyPartialAggregation extends MyAbstractAggregation {

  private long epochSecond;

  public MyPartialAggregation(final long epochSecond) {
    this.epochSecond = epochSecond;
  }

  public void reset(
      long epochSecond
  ) {
    this.epochSecond = epochSecond;
    this.totalDurationInMillis = 0;
    this.numberOfSlowCalls = 0;
    this.numberOfFailedCalls = 0;
    this.numberOfSlowFailedCalls = 0;
    this.numberOfCalls = 0;
  }
}
