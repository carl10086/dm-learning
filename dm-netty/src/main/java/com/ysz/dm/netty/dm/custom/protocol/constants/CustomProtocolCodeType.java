package com.ysz.dm.netty.dm.custom.protocol.constants;

import lombok.Getter;
import lombok.ToString;

@Getter
@ToString
public enum CustomProtocolCodeType {
  V1((byte) 1);

  private final byte val;

  CustomProtocolCodeType(final byte val) {
    this.val = val;
  }

  public static CustomProtocolCodeType of(byte val) {
    for (CustomProtocolCodeType value : CustomProtocolCodeType.values()) {
      if (val == value.getVal()) {
        return value;
      }
    }

    return null;
  }
}
