package com.ysz.dm.fast.basic;

import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.List;

public class Dm_002 {


  public static Thread startAsync(final String url) {
    Thread thread = new Thread(() -> {
      System.out.println("开始下载:" + url);
      try {
        Thread.sleep(100L);
      } catch (InterruptedException ignored) {
      }
      System.out.println("结束下载:" + url);
    });
    thread.start();
    return thread;
  }

  public static void load(Thread t1, Thread t2) throws Exception {
    t1.join();
    t2.join();
    System.out.println("开始执行 load 方法");
    System.out.println("结束执行load 方法");
  }

  public static void main(String[] args) throws Exception {
    load(startAsync("js1"), startAsync("js2"));
    System.in.read();
  }
}
