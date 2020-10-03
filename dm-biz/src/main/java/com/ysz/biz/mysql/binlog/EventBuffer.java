package com.ysz.biz.mysql.binlog;

import com.github.shyiko.mysql.binlog.event.Event;
import java.util.Arrays;
import lombok.Getter;

/**
 * 保留最近的 n 个事件
 */
public class EventBuffer<T> {

  private final int size;

  @Getter
  private final Object[] table;

  @Getter
  private int eventNum = 0;


  public EventBuffer(int size) {
    this.size = size;
    this.table = new Object[size];
  }

  public <T> void add(T event) {
    if (eventNum < size) {
      table[eventNum++] = event;
    } else {
      moveOneStep();
      table[size - 1] = event;
      eventNum++;
    }
  }

  private void moveOneStep() {
    for (int i = 0; i < table.length - 1; i++) {
      table[i] = table[i + 1];
    }
  }


  public static void main(String[] args) {
    EventBuffer<Integer> eventBuffer = new EventBuffer<>(3);
    for (int i = 0; i < 10; i++) {
      eventBuffer.add(i);
      System.err.println(Arrays.toString(eventBuffer.getTable()));
    }
  }
}
