package com.ysz.dm.fast.system.thread.schedule;

import java.util.PriorityQueue;
import lombok.Data;

@Data
public class SchedPriorityItem {

  private PriorityQueue<SchedEntity> entities;


  public SchedEntity top() {
    return null;
  }

  public SchedEntity remove() {
    return null;
  }
}
