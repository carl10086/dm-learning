package com.ysz.dm.fast.basic.juc.ratelimiter;

import com.google.common.util.concurrent.RateLimiter;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class RatelimiterDm_001 {

  public static void main(String[] args) {
    /*1. 默认是基于令牌桶的的算法*/
    final RateLimiter rateLimiter = RateLimiter.create(3);
    rateLimiter.acquire();
    final int calls = 10;
    for (int i = 0; i < calls; i++) {
//      log.error("block:{}", rateLimiter.acquire());
    }
  }

}
