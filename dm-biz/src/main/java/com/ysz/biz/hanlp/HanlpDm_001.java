package com.ysz.biz.hanlp;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.seg.NShort.NShortSegment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.IndexTokenizer;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import java.util.Arrays;
import java.util.List;

public class HanlpDm_001 {

  public static void main(String[] args) throws Exception {
    System.out.println("-----增加自定义词库前----");
    String text = "攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰，习近平";  // 怎么可能噗哈哈！
    /*1. 没有定义词库前*/
//    showAll(text);

    // 动态增加
    CustomDictionary.add("攻城狮", "duitang 1024");
    CustomDictionary.add("狮逆", "duitang 1024");
    CustomDictionary.add("娶白", "duitang 1024");
    CustomDictionary.add("习近平", "duitang 1024");
    // 强行插入
    CustomDictionary.insert("白富美", "duitang 1024");
    // 删除词语（注释掉试试）
//        CustomDictionary.remove("攻城狮");
    System.out.println(CustomDictionary.add("单身狗", "duitang 1024"));
    System.out.println(CustomDictionary.get("单身狗"));
    // 自定义词典在所有分词器中都有效
    /*2. 定义了词库后*/

    System.out.println("-----增加自定义词库后----");

    showAll(text);
  }

  private static void showAll(String text) {
    System.out.println("---start---");
    showNormal(text);
    showNshort(text);
    indexToken(text);
    showCustomOnly(text);
    System.out.println("---end---");
  }


  private static ObjectArrayList<String> parseCustom(String text) {
    final char[] chars = text.toCharArray();
    final ObjectArrayList<String> objectArrayList = new ObjectArrayList<>();
    CustomDictionary.parseText(text,
        (begin, end, value) -> objectArrayList
            .add(new String(chars, begin, end - begin) + "<-" + Arrays
                .toString(value.nature)));
    return objectArrayList;
  }

  public static void showCustomOnly(String text) {
    System.out.println("only cus 结果:" + parseCustom(text));
    final ObjectArrayList<String> objectArrayList = parseCustom(text);
    for (String s : objectArrayList) {
      System.err.println(s);
    }

  }

  private static void showNormal(final String text) {
    final List<Term> segment = HanLP.segment(text);
//    System.out.println("通用分词结果:" + segment);
//    segment.forEach(System.out::println);
  }

  public static void showNshort(final String text) {
    final List<Term> seg = new NShortSegment().enableCustomDictionary(false)
        .enablePlaceRecognize(true)
        .enableOrganizationRecognize(true).seg(text);
    System.out.println("最短路径分词结果:" + seg);
//    seg.forEach(System.err::println);
  }


  public static void indexToken(final String text) {
    final List<Term> segment = IndexTokenizer.segment(text);
//    System.out.println("全 offset 分词:" + segment);
//    segment.forEach(System.out::println);
  }

}
