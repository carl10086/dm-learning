package com.ysz.dm.fast.system.thread.schedule;

import java.util.List;

public class SchedPritoritys {


  private static final int MIN_PRIORITY = 0;
  private static final int MAX_PRIORITY = 0;
  private List<SchedPriorityItem> items;


  public SchedPriorityItem getByLevel(int level) {
    return null;
  }

  public SchedEntity next() {
    for (SchedPriorityItem item : items) {
      // 从 queue 中取第一个 .
      final SchedEntity entity = item.remove();
      if (entity == null) {

      } else {
        if (entity.canExecute()) {
          return entity;
        } else {
          entity.add(item);
        }
      }
    }
    return null;
  }

}
