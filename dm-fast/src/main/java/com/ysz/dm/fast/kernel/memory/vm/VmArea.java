package com.ysz.dm.fast.kernel.memory.vm;

public class VmArea {

  private Long vmStart;
  private Long vmEnd;

  private VmArea vmNext;
  private VmArea vmPrev;

  private Mm vmMm;

  /**
   * 虚拟地址空间区域也维护了一颗红黑树
   */
  private RbNode vmRb;
}
