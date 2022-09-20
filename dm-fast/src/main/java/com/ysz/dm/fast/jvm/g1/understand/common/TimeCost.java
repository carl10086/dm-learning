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
public class TimeCost {

  private final float user;

  private final float sys;

  private final float real;

  public TimeCost(float user, float sys, float real) {
    this.user = user;
    this.sys = sys;
    this.real = real;
  }
}
