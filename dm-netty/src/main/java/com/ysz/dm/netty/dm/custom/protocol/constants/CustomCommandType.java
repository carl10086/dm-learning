package com.ysz.dm.netty.dm.custom.protocol.constants;

import lombok.Getter;
import lombok.ToString;

@Getter
@ToString
public enum CustomCommandType {

  /**
   * 一般的 请求命令
   */
  RPC_REQUEST((byte) 1);

  private final byte val;

  CustomCommandType(final byte val) {
    this.val = val;
  }

  public static CustomCommandType of(byte val) {
    for (CustomCommandType value : CustomCommandType.values()) {
      if (val == value.getVal()) {
        return value;
      }
    }

    return null;
  }
}
