package com.ysz.dm.fast.basic.juc.future;

import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class Future_Java_Dm_002 {


  private void testRunableException() throws Exception {
    ExecutorService executorService = Executors.newFixedThreadPool(1);

    executorService.execute(() -> System.out.println(1 / 0));

    System.in.read();
  }

  private void testCallableException() throws IOException {
    Callable<Integer> callable = () -> {
      try {
        Thread.sleep(1000L);
      } catch (InterruptedException e) {
        System.err.println("捕获到了中断异常");
        return null;
      }
      System.out.println("陈工执行了方法");
      return 1 / 0;
    };

    ExecutorService executorService = Executors.newFixedThreadPool(1);
    Future<Integer> submit = executorService.submit(callable);
    try {
      submit.get(2L, TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      e.printStackTrace();
    } catch (ExecutionException e) {
      e.printStackTrace();
    } catch (TimeoutException e) {
      e.printStackTrace();
    }
  }


  public static void main(String[] args) throws Exception {
    new Future_Java_Dm_002().testCallableException();
  }

}
