package com.ysz.dm.fast.jctools;

import com.google.common.collect.Queues;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;

public class QueuesDm {

  public static void main(String[] args) throws Exception {
    ArrayBlockingQueue<Integer> queue = new ArrayBlockingQueue<Integer>(100);
    for (int i = 0; i < 10; i++) {
      queue.offer(i);

      List<Integer> buffer = new ArrayList<>(10);
    }
  }
}
