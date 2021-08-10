package com.ysz.dm.resilience4j.ratelimiter.config;

import java.time.Duration;
import lombok.Data;

@Data
public class DmRateLimitCfg {

  /**
   * 没有 permit 时的 最大 wait timeout 时间
   */
  private final Duration timeoutDuration;

  /**
   * 定义 逻辑时间单位 、 一个周期
   */
  private final Duration limitRefreshPeriod;

  /**
   *  在一个 周期内允许的最大 限制数
   */
  private final int limitForPeriod;

}
