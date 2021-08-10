package com.ysz.dm.resilience4j.ratelimiter.core;

import com.ysz.dm.resilience4j.ratelimiter.config.DmRateLimitCfg;
import lombok.Data;

@Data
public class DmRateLimiterState {

  /**
   * <pre>
   *   理论上 cfg 是一个不变对象 .
   *
   *   暂时也不需要支持动态改变 ..
   * </pre>
   */
  private DmRateLimitCfg cfg;

  /**
   * 当前 state 对应的 时间周期
   */
  private long activeCycle;

  /**
   * 当前 state 对应的 permits
   */
  private int activePermissions;

  /**
   * <pre>
   *   需要等待的时间 as nano
   * </pre>
   */
  private long nanosToWait;

}
