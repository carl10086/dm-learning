package com.ysz.dm.fast.basic.stream;

import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.stream.Collectors;

public class Java_Stream_Dm_001 {


  public static void main(String[] args) {
    Lists.newArrayList(1, 2, 3, 4).stream().map(x -> x * 2).collect(Collectors.toCollection(
        () -> new ArrayList<>(4)
    ));
  }

}
