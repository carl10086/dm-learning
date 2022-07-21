package com.ysz.dm.fast.algorithm.string.bm.custom;

import it.unimi.dsi.fastutil.chars.Char2IntOpenHashMap;
import java.util.Arrays;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

@Slf4j
public class MyBmV2 {


  /**
   * <pre>
   *   构造一个 hash 结构 用来保存 char 出现的最后一个 index
   * </pre>
   *
   * @param pattern 模式串
   */
  private Char2IntOpenHashMap charLastIndex(char[] pattern) {
    Char2IntOpenHashMap map = new Char2IntOpenHashMap(pattern.length);

    for (int i = 0; i < pattern.length; i++) {
      char c = pattern[i];
      map.put(c, i);
    }

    return map;
  }

  public void generateGS(char[] b, int[] suffix, boolean[] prefix) {
    int m = b.length;
    for (int i = 0; i < m; ++i) { // 初始化
      suffix[i] = -1;
      prefix[i] = false;
    }
    for (int i = 0; i < m - 1; ++i) { // b[0, i]
      int j = i;
      int k = 0; // 公共后缀子串长度
      while (j >= 0 && b[j] == b[m - 1 - k]) { // 与b[0, m-1]求公共后缀子串
        --j;
        ++k;
        suffix[k] = j + 1; //j+1表示公共后缀子串在b[0, i]中的起始下标
      }
      if (j == -1) {
        prefix[k] = true; //如果公共后缀子串也是模式串的前缀子串
      }
    }
  }

  /**
   * <pre>
   *   例子: abcdefg 去找 fg. fg 此时已经移动过一次. f 和 b 对齐的 .
   *
   *   abcdefg
   *    fg
   * </pre>
   *
   * @param source 主串
   * @param pattern 模式串
   * @return 主串中  模式串的第一个位置
   */
  public int search(
      char[] pattern, char[] source, int sourceStart
  ) {

    Char2IntOpenHashMap charLastIndexMap = charLastIndex(pattern);

    /*模式串的长度*/
    int patternLen = pattern.length;

    /* start 就是例子中主串中 c  所在的位置, = 2*/
    int start = sourceStart + patternLen - 1;

    /*主串的最后一个字符位置*/

    while (true) {
      /*当 start > end 的时候. 后面没有字符, 就结束了循环*/
      if (start > source.length - 1) {
        break;
      }
      /*badCharIndex 代表找到坏字符位置, 初始值 <0. 最后还是 <0 则代表没有找到 ;  */
      int badCharIndex = -1;
      for (int i = 0; i < patternLen; i++) {
        int sourceIndex = start - i; // 主串后退 i 格
        int patternIndex = patternLen - 1 - i; // 子串后退 i 格
        /*从 start 开始随着 pattern 往后一个个比, i 每次 +1 , 就往后走一位进行对比*/
        if (source[sourceIndex] /*主串后退*/ == pattern[patternIndex] /*模式串后退*/) {
          continue;
        }
        badCharIndex = sourceIndex;
        break;
      }

      if (badCharIndex < 0) {
        /*<0 表示没有找到坏字符, 意味着找到了, 意味着 ok.*/
        return start - patternLen + 1;
      }

      /*优化: 使用坏字符规则计算移动位数*/
      int calStepByBadChar = getStep(charLastIndexMap, patternLen, source[badCharIndex]);

      log.info("badchar step:{}", calStepByBadChar);

      start = start + calStepByBadChar;
    }
    return -1;
  }

  /**
   * 利用坏字符算法规则计算移动长度
   *
   * @param charLastIndexMap 坏子符使用的 hash 表, 记录了 char 最后一次出现的位置
   * @param patternLen 模式串长度
   * @param badChar 查找到的坏字符
   */
  private int getStep(Char2IntOpenHashMap charLastIndexMap, int patternLen, char badChar) {
    int step = 0;
    if (!charLastIndexMap.containsKey(badChar)) {
      step = patternLen;
    } else {
      step = patternLen - 1 - charLastIndexMap.get(badChar);
    }
    return step;
  }


  public int search(
      String pattern, String source
  ) {
    return this.search(pattern.toCharArray(), source.toCharArray(), 0);
  }


  public static void main(String[] args) throws Exception {
    char[] pattern = "cabcab".toCharArray();
    int[] suffix = new int[pattern.length];
    boolean[] prefix = new boolean[pattern.length];
    new MyBmV2().generateGS(pattern, suffix, prefix);

  }

}