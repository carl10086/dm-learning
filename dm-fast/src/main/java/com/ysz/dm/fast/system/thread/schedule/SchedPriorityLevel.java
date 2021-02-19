package com.ysz.dm.fast.system.thread.schedule;

import org.jetbrains.annotations.NotNull;

public class SchedPriorityLevel implements Comparable<SchedPriorityLevel> {

  private static final int MIN = 0;
  private static final int MAX = 99;
  private int level;

  @Override
  public int compareTo(@NotNull final SchedPriorityLevel o) {
    return this.level - o.level;
  }
}
