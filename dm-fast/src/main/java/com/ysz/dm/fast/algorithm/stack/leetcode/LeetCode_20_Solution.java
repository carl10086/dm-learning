package com.ysz.dm.fast.algorithm.stack.leetcode;


/**
 * https://leetcode-cn.com/problems/valid-parentheses/
 */
public class LeetCode_20_Solution {

  private boolean isValidChar(char aChar) {
    /*什么sb写法*/
    return aChar == '(' || aChar == ')' || aChar == '{' || aChar == '}' || aChar == '['
        || aChar == ']';
  }

  private char findPair(char aChar) {
    if (aChar == '(') {
      return ')';
    }
    if (aChar == ')') {
      return '(';
    }

    if (aChar == '[') {
      return ']';
    }

    if (aChar == ']') {
      return '[';
    }

    if (aChar == '{') {
      return '}';
    }

    if (aChar == '}') {
      return '{';
    }
    throw new IllegalArgumentException("unsupported char:" + aChar);
  }

  public boolean isLeftSide(char aChar) {
    return aChar == '(' || aChar == '{' || aChar == '[';
  }

  public boolean isValid(String s) {
    if (s == null) {
      return false;
    }
    char[] chars = s.toCharArray();
    java.util.LinkedList<Character> stack = new java.util.LinkedList<>();

    for (int i = 0; i < chars.length; i++) {
      char aChar = chars[i];
      if (!isValidChar(aChar)) {
        return false;
      }
      Character peek = stack.peek();
      if (peek == null) {
        /*第一次、直接入栈*/
        if (isLeftSide(aChar)) {
          stack.push(aChar);
          continue;
        } else {
          return false;
        }
      } else {
        if (findPair(peek) == aChar) {
          /*如果刚好凑成了一对, 直接出栈*/
          stack.pop();
        } else {
          if (isLeftSide(aChar)) {
            stack.push(aChar);
          } else {
            return false;
          }
        }
      }
    }
    return stack.isEmpty();
  }

  public static void main(String[] args) {
    System.err.println(new LeetCode_20_Solution().isValid("([])"));
    System.err.println(new LeetCode_20_Solution().isValid("(())"));
    System.err.println(new LeetCode_20_Solution().isValid("()[]{}"));
    System.err.println(new LeetCode_20_Solution().isValid("({[})"));
    System.err.println(new LeetCode_20_Solution().isValid("(){}}{"));
  }
}
