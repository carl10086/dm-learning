package com.ysz.dm.fast.basic.juc.pool;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import java.util.concurrent.Callable;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ThreadPoolBasic_Dm_001 {

  private static class SlowQuery implements Callable<Integer> {

    private final long val;

    private SlowQuery(final long val) {
      this.val = val;
    }

    @Override
    public Integer call() throws Exception {
//      Thread.sleep(val);
      return 1;
    }
  }


  private static ThreadPoolExecutor pool() {
    final int coreSize = 1;
    final int queueSize = 256;
    final int keepAliveTime = 30;
    final String threadNamePrefix = "queryThreadPool-%d";
    return new ThreadPoolExecutor(coreSize, coreSize, keepAliveTime, TimeUnit.MINUTES,
        new LinkedBlockingQueue<>(queueSize),
        new ThreadFactoryBuilder().setNameFormat(threadNamePrefix).build(),
        (r, executor1) -> {
          /*通过自定义的拒绝策略打出更加详细的信息*/
          String msg = String.format("Thread pool is EXHAUSTED!" +
                  " Thread Name: %s, Pool Size: %d (active: %d, core: %d, max: %d, largest: %d), Task: %d (completed: %d), Executor status:(isShutdown:%s, isTerminated:%s, isTerminating:%s)",
              threadNamePrefix
              , executor1.getPoolSize(), executor1.getActiveCount(), executor1.getCorePoolSize(),
              executor1.getMaximumPoolSize(), executor1.getLargestPoolSize(),
              executor1.getTaskCount(), executor1.getCompletedTaskCount(), executor1.isShutdown(),
              executor1.isTerminated(), executor1.isTerminating());
          System.err.println(msg);
//          throw new RejectedExecutionException(msg);
        }
    );
  }

  public static void main(String[] args) throws Exception {
    final ThreadPoolExecutor pool = pool();
    pool.submit(new SlowQuery(10L)).get();
    Thread.sleep(1000L);
    pool.submit(new SlowQuery(10L));
    System.in.read();
  }
}
