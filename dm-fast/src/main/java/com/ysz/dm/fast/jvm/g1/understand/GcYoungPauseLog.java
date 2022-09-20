package com.ysz.dm.fast.jvm.g1.understand;

import com.ysz.dm.fast.jvm.g1.understand.common.MemorySizeChange;
import com.ysz.dm.fast.jvm.g1.understand.common.TimeCost;
import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 *  记录 新生代 gc 的基本信息
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/21
 **/
@ToString
@Getter
public class GcYoungPauseLog {

  private MemorySizeChange eden;
  private MemorySizeChange survivors;
  private MemorySizeChange heap;


  private TimeCost timeCost;

  public GcYoungPauseLog setEden(MemorySizeChange eden) {
    this.eden = eden;
    return this;
  }

  public GcYoungPauseLog setSurvivors(MemorySizeChange survivors) {
    this.survivors = survivors;
    return this;
  }

  public GcYoungPauseLog setHeap(MemorySizeChange heap) {
    this.heap = heap;
    return this;
  }

  public GcYoungPauseLog setTimeCost(TimeCost timeCost) {
    this.timeCost = timeCost;
    return this;
  }
}
