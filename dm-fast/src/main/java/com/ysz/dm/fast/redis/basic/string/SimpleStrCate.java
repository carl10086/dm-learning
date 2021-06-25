package com.ysz.dm.fast.redis.basic.string;

import lombok.Getter;

public enum SimpleStrCate {
  str5(0);

  @Getter
  private final int val;


  SimpleStrCate(final int val) {
    this.val = val;
  }


}
