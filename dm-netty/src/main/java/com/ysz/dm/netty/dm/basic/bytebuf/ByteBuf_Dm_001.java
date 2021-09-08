package com.ysz.dm.netty.dm.basic.bytebuf;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufAllocator;

/**
 * @author carl
 */
public class ByteBuf_Dm_001 {

  public static void main(String[] args) {
    ByteBuf heapBuf = ByteBufAllocator.DEFAULT.heapBuffer(10, 20);
    System.out.println(heapBuf);
    writeBytes(heapBuf, 10);
  }

  private static void writeBytes(ByteBuf heapBuf, int n) {
    for (int i = 0; i < n; i++) {
      heapBuf.writeByte(n);
    }
  }

}
