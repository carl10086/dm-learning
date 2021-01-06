package com.ysz.dm.fast.basic.jol;

import org.openjdk.jol.info.ClassLayout;
import org.openjdk.jol.vm.VM;

public class Jol_Dm_001 {

  public static class A {

    int a;
    int b;
    //    @sun.misc.Contended
    int c;
    int d;
  }

  public static class B {

    Object o;
    int e;
    //    @sun.misc.Contended("first")
    int f;
    //    @sun.misc.Contended("first")
    int g;
    //    @sun.misc.Contended("last")
//    int i;
    //    @sun.misc.Contended("last")
//    int k;
  }

  public static void main(String[] args) {
    System.out.println(VM.current().details());
    System.out.println(ClassLayout.parseClass(B.class).toPrintable());

    B b1 = new B();
    B b2 = new B();
    System.err.println(VM.current().sizeOf(b1));
    System.err.println(VM.current().addressOf(b1));

    System.err.println(VM.current().sizeOf(b2));
    System.err.println(VM.current().addressOf(b2) - VM.current().addressOf(b1));


  }
}
