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
@ToString
@Getter
public class MemorySize {

  private final long used;

  private final long max;

  public MemorySize(long used, long max) {
    this.used = used;
    this.max = max;
  }
}
