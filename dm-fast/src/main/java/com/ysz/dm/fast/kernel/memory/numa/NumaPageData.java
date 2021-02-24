package com.ysz.dm.fast.kernel.memory.numa;


/**
 * 代表的一个存储节点、numa 下有多个节点
 */
public class NumaPageData {

  private static final int MAX_NR_ZONES = calMaxNrZones(); //

  private static int calMaxNrZones() {
    // 复杂的逻辑、这里简单模拟下
    return 2;
  }

  /**
   * 当前节点管理的 zone
   */
  private NumaZone[] nodeZones;

  /**
   * zone 总数
   */
  private int nrZones;

  /**
   * 当前节点中的 第一个 页面 ... , 第一个页面 + 大小就相当于数组了吧 .
   */
  private NumaPage[] nodeMemMap;


  /**
   * 当前节点的 id
   */
  private int nodeId;

  /**
   * 当前节点总共的物理页面数量
   */
  private long nodePresentPages;

  /**
   * 当前节点总共的物理页面数量, 包含空洞
   */
  private long nodeSpannedPages;

  //...


}
