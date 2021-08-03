package com.ysz.biz.hanlp;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.dictionary.py.Pinyin;
import java.lang.Character.UnicodeScript;
import java.util.List;

public class HanlpDm_pinyin_002 {

  public static boolean isChinese(char c) {
    Character.UnicodeBlock ub = Character.UnicodeBlock.of(c);
    if (ub == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS ||
        ub == Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS ||
        ub == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A ||
        ub == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B ||
        ub == Character.UnicodeBlock.CJK_SYMBOLS_AND_PUNCTUATION ||
        ub == Character.UnicodeBlock.HALFWIDTH_AND_FULLWIDTH_FORMS ||
        ub == Character.UnicodeBlock.GENERAL_PUNCTUATION) {
      return true;
    }
    return false;
  }

  private static String preHandle(String source) {
    if (source == null) {
      return null;
    }
    char[] chars = source.toCharArray();
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < chars.length; i++) {
      char aChar = chars[i];
      if ((aChar >= 'a' && aChar <= 'z') || (aChar >= 'A' && aChar <= 'Z')) {
        sb.append(aChar);
      } else {
        int codePoint = source.codePointAt(i);
        if (Character.UnicodeScript.of(codePoint) == UnicodeScript.HAN) {
          sb.append(aChar);
        }
      }
    }
    return sb.toString();
  }

  public static void main(String[] args) {
//    System.out.println(preHandle("最终发现了所有吸&&&fdsf尽平"));
    tst();
  }

  private static void tst() {
    String word = "习近平";
    CustomDictionary.insert(
        "xijinping", "dt 1024"
    );

    String text = "最终发现了所有吸&&&fdsf尽平";
    List<Pinyin> pinyinList = HanLP.convertToPinyinList(text);

    StringBuilder newLine = new StringBuilder();
    for (Pinyin pinyin : pinyinList) {
      String tone = pinyin.getPinyinWithoutTone();
      if (!"none".equalsIgnoreCase(tone)) {
        newLine.append(tone);
      }else {
        System.err.println(1);
      }
    }

    System.out.println(HanlpDm_001.parseCustom(
        newLine.toString()
    ));
  }

}
