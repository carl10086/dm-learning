package com.ysz.dm.fast.algorithm.string.bm;

import it.unimi.dsi.fastutil.chars.Char2IntMap;
import it.unimi.dsi.fastutil.chars.Char2IntOpenHashMap;

public class BmTmp {


  /**
   * 记录每个字符最后出现的位置
   */
  private Char2IntMap bc(char[] array) {
    Char2IntMap map = new Char2IntOpenHashMap();
    for (int i = 0; i < array.length; i++) {
      map.put(array[i], i);
    }

    return map;
  }


  public int execute(String pattern, String target) {

    char[] a = target.toCharArray();
    int n = a.length;

    char[] b = pattern.toCharArray();
    int m = b.length;

    /*构建 hash, 记录模式串最后出现的位置*/
    Char2IntMap bc = bc(b);

    /*表示主串和模式串对齐的第1个字符*/
    int i = 0;

    while (i < n - m) {
      int j;

      for (j = m - 1; j >= 0; --j) { /*模式串从后前匹配*/
        if (a[i + j] != b[j]) {
          break;
        }
      }

      if (j < 0) {
        return i; /* <0 说明 没有任何坏字符, 匹配成功, 返回主串和模式串迪哥匹配字符的位置*/
      }

      /*等同于 模式串往后滑动位数*/
      i = i + (j - (bc.containsKey(a[i + j]) ? (bc.get(a[i + j])) : -1));
    }

    return -1;

  }


  public static void main(String[] args) {
    String pattern = "abd";
    String target = "abcacabdc";

    int execute = new BmTmp().execute(pattern, target);
    System.out.println(execute);
  }


}
