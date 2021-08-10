package com.ysz.dm.fast.basic.juc.executor;

import com.google.common.base.Joiner;
import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Thread_Joiner_Dm_02 {

  public void execute() throws Exception {
    final ExecutorService executorService = Executors.newFixedThreadPool(10);
    Thread_Joiner_Dm_02 dm = Thread_Joiner_Dm_02.this;
    SimpleBatchTools.batchWithLargeFile(new File("/Users/carl/tmp/fuck/3.txt"), 5,
        strings -> executorService.submit(new Task(strings, dm)));
    System.in.read();
  }

  public void doWithUserIds(List<String> userIds) {
    final String join = Joiner.on(",").join(userIds);
    log.warn("{},str:{}", userIds, join);
    try {
      Thread.sleep(10L);
    } catch (InterruptedException e) {
    }
  }


  public static void main(String[] args) throws Exception {
    new Thread_Joiner_Dm_02().execute();
  }


  private static class Task implements Runnable {

    private final List<String> userIds;
    private final Thread_Joiner_Dm_02 worker;

    private Task(
        final List<String> userIds,
        final Thread_Joiner_Dm_02 worker) {
      this.userIds = userIds;
      this.worker = worker;
    }

    @Override
    public void run() {
      this.worker.doWithUserIds(userIds);
    }
  }

}
