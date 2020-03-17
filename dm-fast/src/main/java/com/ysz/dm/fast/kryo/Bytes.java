package com.ysz.dm.fast.kryo;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
public class Bytes {

  public static byte[] int2bytes(int v) {
    byte[] b = new byte[]{0, 0, 0, 0};
    int off = 0;
    b[off + 3] = (byte) v;
    b[off + 2] = (byte) (v >>> 8);
    b[off + 1] = (byte) (v >>> 16);
    b[off + 0] = (byte) (v >>> 24);
    return b;
  }

  public static int bytes2int(byte[] b, int off) {
    return ((b[off + 3] & 255) << 0) + ((b[off + 2] & 255) << 8) + ((b[off + 1] & 255) << 16) + (
        b[off + 0] << 24);
  }


}
