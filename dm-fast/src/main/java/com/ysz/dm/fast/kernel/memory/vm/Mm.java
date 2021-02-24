package com.ysz.dm.fast.kernel.memory.vm;

import java.util.LinkedList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class Mm {

  private LinkedList<VmArea> mmap; // 虚拟地址空间列表
  private RbRoot mmRb; // 用来虚拟地址空间查询的红黑树
  private Pgd pgd; // pgd 指针, page global directory
  private AtomicInteger mmUsers; // 访问用户空间的总用户数
  private AtomicInteger mmCount; // 用户使用计数器
  private AtomicLong nrPtes; // 页表的页面总数

  private int mapCount; // 正在被使用的  VMA 数量
  private LinkedList<Mm> mmList; // 所有的 mmstruct 都通过它 链接

  private Long totalVm; // 所有的 vma 区域 + 起来的内存综合
  private Long startCode; // 代码段起始地址
  private Long endCode; // 代码段结束地址

  private Long startData; // 数据段起始地址
  private Long endData; // 数据段结束地址

  private Long startBrk; // 堆的起始地址
  private Long brk; // 堆的结束地址

  private Long startStack; // 栈的起始地址
  private Long argStart;
  private Long argEnd;

  private Long envStart;
  private Long envEnd;


}
