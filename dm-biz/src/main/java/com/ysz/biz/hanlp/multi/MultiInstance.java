package com.ysz.biz.hanlp.multi;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.HanLP.Config;
import com.hankcs.hanlp.dictionary.DynamicCustomDictionary;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import java.util.List;

/**
 * <per>
 *   参考: https://github.com/hankcs/HanLP/issues/1339
 * </per>
 */
public class MultiInstance {

  public static void main(String[] args) {
    final DynamicCustomDictionary d1 = new DynamicCustomDictionary(Config.CustomDictionaryPath);
    final DynamicCustomDictionary d2 = new DynamicCustomDictionary(Config.CustomDictionaryPath);

    showCustomOnly("自定义测试当时方式电风扇", d1);

    d1.insert("自定义测试", "dt 1024");
    showCustomOnly("自定义测试当时方式电风扇", d1);
    showCustomOnly("自定义测试当时方式电风扇", d2);

  }


  public static void showCustomOnly(String text, DynamicCustomDictionary dictionary) {
    System.err.println("--------------------");
    final Segment segment = HanLP.newSegment();
    segment.customDictionary = dictionary;
    final List<Term> seg = segment.seg(text);
    seg.forEach(System.err::println);
  }
}
