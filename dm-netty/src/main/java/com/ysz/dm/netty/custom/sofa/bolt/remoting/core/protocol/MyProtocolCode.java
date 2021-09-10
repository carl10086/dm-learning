package com.ysz.dm.netty.custom.sofa.bolt.remoting.core.protocol;

import com.alipay.remoting.ProtocolCode;
import java.util.Arrays;

public class MyProtocolCode {

  byte[] version;

  private MyProtocolCode(byte[] version) {
    this.version = version;
  }

  public static MyProtocolCode fromBytes(byte... version) {
    return new MyProtocolCode(version);
  }


  public byte getFirstByte() {
    return this.version[0];
  }

  public int length() {
    return this.version.length;
  }


  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("MyProtocolCode{");
    sb.append("version=").append(Arrays.toString(version));
    sb.append('}');
    return sb.toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    MyProtocolCode that = (MyProtocolCode) o;

    return Arrays.equals(version, that.version);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(version);
  }
}
