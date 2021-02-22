package com.ysz.dm.fast.kernel.memory.numa;

/**
 * 一个存储节点下 包含了 ZONE_DMA, ZONE_NORMAL, ZONE_HIGHMEM 三个管理区
 *
 * - ZONE_DMA: 0-16M, 这个区域的页面专门为 提供 IO 设备的  DMA 使用 .
 * - ZONE_NORMAL: 16M - 896M, 内核能够直接使用的
 * - ZONE_HIGHMEM: 896M - end 高端内存、内核不能直接使用
 *
 */
public class NumaZone {

  /**
   * 对应管理节点的 id
   */
  int node;

  private NumaPageData zonePagdat;

  /**
   * 保存内存区域的惯用名称
   * NORMAL
   * DMA
   * HIGHMEM
   */
  private String name;

  private long zoneStartPfn;

}
