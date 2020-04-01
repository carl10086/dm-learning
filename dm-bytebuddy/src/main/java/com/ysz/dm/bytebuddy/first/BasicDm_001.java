package com.ysz.dm.bytebuddy.first;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.implementation.FixedValue;
import net.bytebuddy.matcher.ElementMatchers;

public class BasicDm_001 {


  public static void main(String[] args) throws Exception {
    /*重写 Object 对象的 toString 方法*/
    Class<?> toStringProxyClass = new ByteBuddy()
        /*指定父类*/
        .subclass(Object.class)
        /*实现 matchers 机制来匹配方法*/
        .method(ElementMatchers.named("toString"))
        /*利用一个 Implementation 来实现表示一个动态定义的方法*/
        .intercept(FixedValue.value("Hello ByteBuddy"))
        .make()
        /*指定类加载器*/
        .load(BasicDm_001.class.getClassLoader())
        .getLoaded();
    System.out.println(toStringProxyClass.newInstance().toString());
  }
}
