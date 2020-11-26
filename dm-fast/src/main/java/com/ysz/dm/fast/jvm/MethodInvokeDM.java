package com.ysz.dm.fast.jvm;

public class MethodInvokeDM {

  private void m1() {

  }

  public void tst() {
    TypicalObject.staticAdd(1);
    m1();
  }


}
