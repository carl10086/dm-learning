package com.ysz.dm.fast.kernel.thread.schedule;

/**
 * 每一个 cpu 对应的运行队列
 */
public class SchedRq {

  /**
   * 表示当前队列中有多少可以运行的任务 .
   *
   * 包含了所有的 sched class 任务 .
   */
  private int nrRunning;

  /**
   * 嵌套的 cfs 调度器运行队列
   */
  private SchedCfsRq cfs;

  /**
   * 实时任务调度器运行队列
   */
  private SchedRtRq rt;

  /**
   * 当前正在运行的进程描述符
   */
  private ProcessTaskStruct curr;
}
