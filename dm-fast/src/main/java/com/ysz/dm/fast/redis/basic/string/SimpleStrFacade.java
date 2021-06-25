package com.ysz.dm.fast.redis.basic.string;

import com.google.common.base.Preconditions;

public class SimpleStrFacade {

  public static SimpleStr init(int len) {
    Preconditions.checkArgument(len > 0);
    if (len < (2 ^ 5)) {
      return new SimpleStr5(len);
    }

    throw new RuntimeException("not support");
  }

}
