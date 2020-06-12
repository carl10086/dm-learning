package com.ysz.dm.fast.basic;

import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.List;

public class Dm_002 {


  private static void test(List<String> data) {
    int finalSize = (int) Math.pow(2, data.size());
    List<List<Boolean>> result = new ArrayList<>(finalSize);
    int size = data.size();
    List<Boolean> l1 = new ArrayList<>(size);
    l1.add(true);
    List<Boolean> l2 = new ArrayList<>(size);
    l2.add(false);

    result.add(l1);
    result.add(l2);

    for (int i = 1; i < data.size(); i++) {
      List<List<Boolean>> copyResult = new ArrayList<>();
      for (List<Boolean> booleans : result) {
        List<Boolean> cloneData = new ArrayList<>(booleans);
        copyResult.add(cloneData);
      }
      copyResult.forEach(x -> x.add(true));
      result.forEach(x -> x.add(false));
      result.addAll(copyResult);

    }

    System.err.println(result.size());
    for (List<Boolean> booleans : result) {
      System.out.println(booleans);
    }

  }


  public static void main(String[] args) {
    test(Lists.newArrayList(
        "p3", "p4", "p5", "p6"
    ));


  }

}
