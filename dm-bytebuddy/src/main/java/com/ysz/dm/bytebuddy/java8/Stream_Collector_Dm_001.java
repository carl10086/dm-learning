package com.ysz.dm.bytebuddy.java8;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author carl
 */
public class Stream_Collector_Dm_001 {


  public static void main(String[] args) {
    ArrayList<Integer> collect = Stream.of(1, 2, 3).map(x -> x * 2).collect(Collectors.toCollection(
        () -> new ArrayList<>(3)
    ));
    System.out.println(collect);
  }


}
