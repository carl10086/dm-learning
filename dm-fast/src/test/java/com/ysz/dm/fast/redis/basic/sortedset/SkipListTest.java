package com.ysz.dm.fast.redis.basic.sortedset;

import org.junit.Test;

public class SkipListTest {

  @Test
  public void search() throws Exception {
    final SkipList skipList = new SkipList();

    skipList.add(10);

    skipList.getHeads().forEach(
        System.out::println
    );
  }
}