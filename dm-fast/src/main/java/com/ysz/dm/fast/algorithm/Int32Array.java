package com.ysz.dm.fast.algorithm;

public class Int32Array {

  private Int32[] data;


  public Int32Array of(int length) {
    Int32Array int32Array = new Int32Array();
    int32Array.data = new Int32[length];
    return int32Array;
  }
}
