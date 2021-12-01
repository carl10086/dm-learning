package com.ysz.dm.fast.basic.juc.lock;

import com.google.common.util.concurrent.Striped;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.Lock;

public class LockStripedDm_002 {

  private static Striped<Lock> striped = Striped.lock(128);


  public static void main(String[] args) {
//    Striped.lock();
    System.out.println(getLock("AaAa"));
    System.out.println(getLock("AaAa"));
    System.out.println(getLock("BBBB"));
    System.out.println(getLock("CCCCC"));

    Map<String, String> map = new HashMap<>();

    map.put("AaAa", "AaAa");
    map.put("BBBB", "BBBB");

    System.out.println(map);
  }


  public static Lock getLock(String key) {
    return striped.get(key);
  }

}
