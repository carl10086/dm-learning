package com.ysz.dm.fast.algorithm;

import lombok.Getter;

@Getter
public final class ArrayBuffer {

  private final byte[] bytes;

  private final int offset;

  private final int length;


  public ArrayBuffer(byte[] bytes) {
    this.bytes = bytes;
    this.offset = 0;
    this.length = bytes.length;
  }

  public ArrayBuffer(int length) {
    this.bytes = new byte[length];
    this.length = length;
    this.offset = 0;
  }
}
