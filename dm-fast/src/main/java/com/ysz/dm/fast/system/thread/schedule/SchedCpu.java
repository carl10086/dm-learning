package com.ysz.dm.fast.system.thread.schedule;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 代表一个 cpu;
 * 系统的门面对象;
 */
public class SchedCpu {

  private AtomicBoolean running;

  private final int cpuId;

  private final SchedPritoritys pritoritys;

  public static SchedCpu newCpu(final int cpuId) {
    SchedPritoritys pritoritys = new SchedPritoritys();
    SchedCpu cpu = new SchedCpu(
        cpuId, pritoritys
    );
    return cpu;
  }

  public SchedCpu(final int cpuId, final SchedPritoritys pritoritys) {
    this.cpuId = cpuId;
    this.running = new AtomicBoolean(true);
    this.pritoritys = pritoritys;
  }


  /**
   * 触发调度
   */
  public void schedule() {

  }

}
