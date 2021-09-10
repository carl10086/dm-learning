package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol.version;

import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public enum MyProtocolVersion {

  V1((byte) 1), V2((byte) 2);

  private final byte val;

  MyProtocolVersion(final byte val) {
    this.val = val;
  }

  public static MyProtocolVersion of(byte val) {

    for (MyProtocolVersion version : MyProtocolVersion.values()) {
      if (version.getVal() == val) {
        return version;
      }
    }

    return null;
  }


  public static byte defaultVersion() {
    return V2.getVal();
  }
}
