package com.ysz.dm.fast.basic.juc.executor;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * @author carl
 */
public class ThreadExecutor_Dm_001 {

  public static void main(String[] args) throws Exception {
    final ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
        1,
        4,
        10L,
        TimeUnit.MINUTES,
        new ArrayBlockingQueue<>(3),
        new ThreadFactoryBuilder().setNameFormat("testPool" + "-$d").build(),
        new RejectedExecutionHandler() {
          @Override
          public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
            System.err.printf("线程打印拒绝日志:%s -> %s", Thread.currentThread().getName(), r.toString());
          }
        }
    ) {
      @Override
      protected void afterExecute(Runnable r, Throwable t) {
        super.afterExecute(r, t);
        if (t != null) {
          System.err.println("捕获到子线程的异常:");
          t.printStackTrace();
        }
      }

    };

    Thread thread = new Thread(new ThreadExecutorMonitor(threadPoolExecutor));
    thread.setDaemon(true);
    thread.start();

    for (int j = 0; j < 10; j++) {
      threadPoolExecutor.execute(() -> {
        /*这里暂时无所谓、一个类死循环*/
        for (int i = 0; i < 100; i++) {
          try {
            Thread.sleep(1000L);
            System.out.println(1 / 0);
          } catch (InterruptedException ignored) {
          }
        }
      });
      /*每隔2s 加入一个任务*/
      Thread.sleep(2000L);
    }

    threadPoolExecutor.shutdown();
  }

}
