package com.ysz.dm.fast.basic.juc.future;


import io.vavr.concurrent.Future;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import lombok.extern.slf4j.Slf4j;

/**
 * @author carl
 */
@Slf4j
public class Future_Dm_001 {


  private final Executor executor = Executors.newFixedThreadPool(2);

  public void test2() throws Exception {

  }

  public void test() throws Exception {
    Future<Integer> of = Future.of(executor, this::slowMethod);
    Thread.sleep(2L);
    System.out.println("开始");
    System.out.println(
        Future.of(executor, this::fastMethod).await(1500L, TimeUnit.MILLISECONDS).isSuccess());
    System.out.println(of.await(1001L, TimeUnit.MILLISECONDS).isSuccess());
  }

  public int fastMethod() {
    try {
      System.out.println("fast Method:" + Thread.currentThread().getName());
      Thread.sleep(500L);
    } catch (Exception e) {
      System.err.println("监控到了中断异常");
    }
    return 1;
  }


  public int slowMethod() {
    try {
      System.out.println("slow Method");
      Thread.sleep(1500L);
    } catch (Exception e) {
      System.err.println("监控到了中断异常");
    }
    return 0;
  }

  public static void main(String[] args) throws Exception {
    new Future_Dm_001().test();
  }

}
