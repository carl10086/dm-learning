package com.ysz.dm.resilience4j.circuit.dm;

import io.github.resilience4j.core.IntervalFunction;

public class Circuit_Dm_IntervalFunction_Dm_003 {

  public void execute() throws Exception {
    /*貌似本质上就是利用  function 的 迭代实现的指数功能. Stream.iterate ..*/
    final IntervalFunction intervalFunction = IntervalFunction.ofExponentialBackoff(
        10L,
        2.0
    );

    System.err.println(intervalFunction.apply(1));
    System.err.println(intervalFunction.apply(2));
    System.err.println(intervalFunction.apply(3));

  }

  public static void main(String[] args) throws Exception {
    new Circuit_Dm_IntervalFunction_Dm_003().execute();
  }
}
