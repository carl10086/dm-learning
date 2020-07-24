package com.ysz.dm.bytebuddy.jvm;

/**
 * @author carl
 */
public class DmBeanImpl implements DmBean {


  private void invokeSpecial() {
    System.out.println("invokeSpecial");
  }

  public static void invokeStatic() {
    System.out.println("invokeStatic");
  }

  public final void invokeVirtual() {
    System.out.println("invokeVirtual");
  }

  @Override
  public void invokeInterface() {
    System.out.println("invokeInterface");
  }

  public void test() {
    this.invokeSpecial();
    this.invokeVirtual();
    invokeStatic();
  }
}
