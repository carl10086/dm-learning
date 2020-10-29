package com.ysz.dm.fast.basic.jol;

import org.openjdk.jol.info.ClassLayout;
import org.openjdk.jol.vm.VM;

public class Jol_Dm_001 {

  public static class A {

    int a;
    int b;
    @sun.misc.Contended
    int c;
    int d;
  }

  public static class B extends A {

    int e;
    @sun.misc.Contended("first")
    int f;
    @sun.misc.Contended("first")
    int g;
    @sun.misc.Contended("last")
    int i;
    @sun.misc.Contended("last")
    int k;
  }

  public static void main(String[] args) {
    System.out.println(VM.current().details());
    System.out.println(ClassLayout.parseClass(B.class).toPrintable());
  }
}
