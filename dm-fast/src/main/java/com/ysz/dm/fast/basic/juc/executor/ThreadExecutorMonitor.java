package com.ysz.dm.fast.basic.juc.executor;

import com.google.common.base.Joiner;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

/**
 * @author carl
 */
public class ThreadExecutorMonitor implements Runnable {


  public final ThreadPoolExecutor executor;

  private final int intervalSecs = 1;

  public ThreadExecutorMonitor(ThreadPoolExecutor executor) {
    this.executor = executor;
  }


  private String executorInfo() {
    Map<String, Object> info = new HashMap<>();
    /*近似的线程数*/
    info.put("activeCnt", executor.getActiveCount());
//    info.put("corePoolSize", executor.getCorePoolSize());
//    info.put("maxPoolSize", executor.getMaximumPoolSize());
    /*当前近似的总任务数*/
    info.put("taskCount", executor.getTaskCount());
//    info.put("largetestPoolSize", executor.getLargestPoolSize());
    /*会阻塞的去获取 队列的长度*/
    info.put("queueSize", executor.getQueue().size());

    return String.format("%s",
        Joiner.on(",").join(
            info.entrySet().stream()
                .map(x -> String.format("%s->%s", x.getKey(), Objects.toString(x.getValue())))
                .collect(Collectors.toList())
        ));
  }

  @Override
  public void run() {
    while (true) {
      try {
        System.out.println(executorInfo());
        Thread.sleep(1 * 1000L);
      } catch (InterruptedException ignored) {
        break;
      }
    }
  }
}
