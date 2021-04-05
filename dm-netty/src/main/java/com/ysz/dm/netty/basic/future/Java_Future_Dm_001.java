package com.ysz.dm.netty.basic.future;

import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;

public class Java_Future_Dm_001 {

  public static void main(String[] args) throws Exception {
    Callable<String> callable = new Callable<String>() {
      @Override
      public String call() throws Exception {
        Thread.sleep(10L);
        return "ok";
      }
    };

    FutureTask<String> futureTask = new FutureTask<>(callable);
    final Thread thread = new Thread(futureTask);
    thread.start();
    final String s = futureTask.get();
    System.err.println(s);
  }

}
