package com.ysz.dm.fast.basic.onjava.features;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/27
 **/
public class SwitchNewFeatures {

  /**
   * patterns are not support in switch in java17
   */
  static void combineNullAndCase(String s) {
    switch (s) {
//      case "XX", null -> System.out.println("XX|null");
      default -> System.out.println("default");
    }
  }

  /**
   * yield 关键字用 switch 中直接返回
   */
  static int colon(String s) {
    var result = switch (s) {
      case "i":
        yield 1;
      case "j":
        yield 2;
      case "k":
        yield 3;
      default:
        yield 0;
    };
    return result;
  }

  /**
   * 使用箭头是和 yield 相同
   */
  static int arrow(String s) {
    var result = switch (s) {
      case "i" -> 1;
      case "j" -> 2;
      case "k" -> 3;
      default -> 0;
    };
    return result;
  }
}
