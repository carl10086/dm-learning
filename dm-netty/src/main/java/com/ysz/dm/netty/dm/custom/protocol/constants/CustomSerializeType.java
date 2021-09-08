package com.ysz.dm.netty.dm.custom.protocol.constants;

import lombok.Getter;
import lombok.ToString;

/**
 * body 使用的序列化方式
 */
@ToString
@Getter
public enum CustomSerializeType {
  json((byte) 1);

  private final byte val;

  CustomSerializeType(final byte val) {
    this.val = val;
  }

  public static CustomSerializeType of(byte val) {
    for (CustomSerializeType value : CustomSerializeType.values()) {
      if (val == value.getVal()) {
        return value;
      }
    }

    return null;
  }
}
