package com.ysz.dm.fast.basic.juc.threadlocal;

import org.apache.log4j.helpers.ThreadLocalMap;

public class ThreadLocalDm_001 {

  public static void main(String[] args) throws Exception {

    Thread thread = new Thread(() -> {
      ThreadLocal<String> threadLocal = new ThreadLocal<>();
      threadLocal.set("1");
      threadLocal.set("2");
    });
    thread.start();
    System.in.read();
  }

}
