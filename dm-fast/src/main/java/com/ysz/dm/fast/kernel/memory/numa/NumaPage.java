package com.ysz.dm.fast.kernel.memory.numa;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * 代表一个物理页面、 Linux 的中间层吗?
 *
 * 内核用数据结构描述一个 页框的状态信息 . 所有的页描述符存放在 pagedata 的 node_mem_map 数组中, 数组的下标是 页框号 ;
 */
public class NumaPage {

  private Long flags; // 这些标志用来描述页面的状态
  private AtomicInteger _count; // 页面使用计数器
  private NumaAddressSpace mappings; // 物理页映射的线性地址空间 , 没记错的话，其中封装了 内核高速缓存逻辑
}
