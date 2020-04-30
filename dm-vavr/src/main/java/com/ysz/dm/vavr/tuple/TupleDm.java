package com.ysz.dm.vavr.tuple;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import org.junit.Test;

public class TupleDm {

  /**
   * 可以使用 Map 进行 Tuple 的转换
   */
  @Test
  public void tstMap() {
    Tuple2<String, Integer> java8 = Tuple.of("java", 8);
    System.err.println(java8.map((s, i) -> Tuple.of(s.substring(2) + "vr", i / 8)));
  }

  @Test
  public void tstApply() {
    Tuple2<String, Integer> java8 = Tuple.of("java", 8);
    String apply = java8.apply(
        (s, i) -> s.substring(2) + "vr " + i / 8
    );
    System.err.println(apply);
  }

}
