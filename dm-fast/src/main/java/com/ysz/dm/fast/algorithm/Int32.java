package com.ysz.dm.fast.algorithm;

import org.apache.hadoop.hbase.util.Bytes;

public class Int32 {

  private final ArrayBuffer arrayBuffer;


  public Int32(ArrayBuffer arrayBuffer) {
    this.arrayBuffer = arrayBuffer;
  }

  public void add(Int32 int32) {
    /*默认*/
    int src = Bytes.toInt(int32.arrayBuffer.getBytes());
    int dest = Bytes.toInt(arrayBuffer.getBytes());
    System.out.println(src + dest);
  }

  public static void main(String[] args) {
    Int32 int32 = new Int32(new ArrayBuffer(Bytes.toBytes(300)));
    // 514
    int32.add(new Int32(new ArrayBuffer(new byte[]{0, 0, 2, 2})));
  }
}
