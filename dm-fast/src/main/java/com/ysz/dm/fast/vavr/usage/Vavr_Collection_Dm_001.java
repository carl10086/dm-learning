package com.ysz.dm.fast.vavr.usage;

import com.google.common.collect.Lists;
import io.vavr.collection.List;

/**
 * @author carl
 */
public class Vavr_Collection_Dm_001 {


  public static void main(String[] args) {
    List<Integer> data = List.ofAll(Lists.newArrayList(1, 2, 3, 4));
    List<Integer> map = data.map(x -> x * 2);
  }


}
