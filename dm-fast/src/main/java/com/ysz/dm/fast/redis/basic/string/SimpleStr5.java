package com.ysz.dm.fast.redis.basic.string;

import com.google.common.base.Preconditions;

/**
 * 2^5 长度内的字符串 ......
 */
public class SimpleStr5 implements SimpleStr {

  @Override
  public SimpleStrCate cate() {
    return SimpleStrCate.str5;
  }

  @Override
  public int total() {
    return 0;
  }

  @Override
  public int used() {
    return 0;
  }

  /**
   * 标记位
   */
  private Byte flags;

  /**
   * 存储的 buf ....
   */
  private char[] buf;

  public SimpleStr5(int initLen) {
    Preconditions.checkState(initLen <= 31, "len 必须 31 个字符内");
    /*cate 是 Byte 的前3 bit*/
    byte firstPart = (byte) cate().getVal();
    /*initLen 是 Byte 的后5 bit*/
    byte secondPart = (byte) initLen;

    this.flags = (byte) ((firstPart << 5) ^ secondPart);
    this.buf = new char[initLen];
  }

  public static int lowerBits(byte b, int len) {
    return b >> (8 - len);
  }

  public static int higherBits(byte b, int len) {
    byte mv = (byte) (0xff >> (8 - len));
    return (b & mv);
  }

  /**
   * debug 用、打印二进制字符串
   */
  private static String asBinaryStr(int b) {
    String finalStr = "";
    String s = Integer.toBinaryString(b);
    if (s.length() >= 8) {
      s = finalStr;
    } else {
      int remain = 8 - s.length();
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < remain; i++) {
        sb.append("0");
      }

      s = sb.append(s).toString();
    }
    return s;

  }
}
