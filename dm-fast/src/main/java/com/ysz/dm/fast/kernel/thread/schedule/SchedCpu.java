package com.ysz.dm.fast.kernel.thread.schedule;

/**
 * 代表一个 调度中的一个 cpu;
 * 系统的门面对象;
 */
public class SchedCpu {

  /**
   * cpu 唯一编号 ...
   */
  private int cpuId;

  /**
   * 一个 cpu 对应一个 运行队列 ...
   */
  private SchedRq rq;


}
