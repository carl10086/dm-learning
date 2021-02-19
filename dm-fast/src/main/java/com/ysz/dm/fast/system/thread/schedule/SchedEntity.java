package com.ysz.dm.fast.system.thread.schedule;

import lombok.Data;
import org.jetbrains.annotations.NotNull;

/**
 * 代表被调度的实体
 */
@Data
public class SchedEntity implements Comparable<SchedEntity> {

  /**
   * 暂时不约束、仅仅用简单的 int 表示把
   */
  private int entityId;

  /**
   * 优先级别
   */
  private SchedPriorityLevel level;


  @Override
  public int compareTo(@NotNull final SchedEntity o) {
    return this.level.compareTo(o.level);
  }

  public boolean canExecute() {
    return false;
  }

  public void add(final SchedPriorityItem item) {
  }
}
