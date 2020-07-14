package com.ysz.dm.fast.jctools;

import java.util.ArrayList;
import java.util.List;
import org.jctools.queues.SpscArrayQueue;

public class JctooldDm_001 {

  public static void main(String[] args) {
    SpscArrayQueue<Integer> queue = new SpscArrayQueue<>(10);
    for (int i = 0; i < 10; i++) {
      System.out.println(queue.offer(i));
    }

    List<Integer> buffer = new ArrayList<>();
    queue.drain(buffer::add);
    for (Integer i : buffer) {
      System.out.println(i);
    }
  }
}
