package com.ysz.dm.fast.basic.onjava.base;

import java.util.Random;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/26
 **/
public class Enums {

  private static Random rand = new Random(47);

  public static <T extends Enum<T>> T random(Class<T> ec) {
    return random(ec.getEnumConstants());
  }

  public static <T> T random(T[] values) {
    return values[rand.nextInt(values.length)];
  }
}
