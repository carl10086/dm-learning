package com.ysz.dm.fast.kernel.thread.schedule;

import lombok.Data;
import org.jetbrains.annotations.NotNull;

/**
 * 代表被调度的实体
 */
@Data
public class SchedEntity implements Comparable<SchedEntity> {



  @Override
  public int compareTo(@NotNull final SchedEntity o) {
    return 0;
  }


}
