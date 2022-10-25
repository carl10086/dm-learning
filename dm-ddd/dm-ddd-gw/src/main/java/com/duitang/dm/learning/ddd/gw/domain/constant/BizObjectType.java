package com.duitang.dm.learning.ddd.gw.domain.constant;

import lombok.Getter;

@Getter
public enum BizObjectType {
  atlas((byte) 0);

  private final byte val;


  BizObjectType(byte val) {
    this.val = val;
  }


  public static BizObjectType fromVal(byte val) {
    for (BizObjectType value : BizObjectType.values()) {
      if (value.val() == val) {
        return value;
      }
    }

    return null;
  }

}
