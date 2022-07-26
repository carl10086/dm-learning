package com.duitang.dm.learning.ddd.gw.infra.core;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class BaseCnt {

  /**
   * 最后一次的时间
   */
  private Instant at;

  private int cnt;

  public BaseCnt setAt(Instant at) {
    this.at = at;
    return this;
  }

  public BaseCnt setCnt(int cnt) {
    this.cnt = cnt;
    return this;
  }
}
