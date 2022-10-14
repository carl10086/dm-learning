package com.ysz.dm.fast.jvm.g1.understand.common;

import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/21
 **/
@Getter
@ToString
public class MemorySizeChange {

  private final MemorySize before;

  private final MemorySize after;

  public MemorySizeChange(MemorySize before, MemorySize after) {
    this.before = before;
    this.after = after;
  }
}
