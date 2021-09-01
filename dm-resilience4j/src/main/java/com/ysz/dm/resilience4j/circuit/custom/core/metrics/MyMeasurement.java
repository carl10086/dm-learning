package com.ysz.dm.resilience4j.circuit.custom.core.metrics;

public class MyMeasurement extends MyAbstractAggregation {

  public void reset() {
    this.totalDurationInMillis = 0;
    this.numberOfSlowCalls = 0;
    this.numberOfFailedCalls = 0;
    this.numberOfCalls = 0;
    /*bugs or useless .to be confirmed*/
//    this.numberOfSlowFailedCalls = 0;
  }
}
