package com.ysz.biz.hanlp.strategy;

import com.google.common.collect.Lists;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.HanLP.Config;
import com.hankcs.hanlp.dictionary.DynamicCustomDictionary;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import java.util.List;

public class MultiStrategy {

  private List<String> dicts;
  private DynamicCustomDictionary dictionary;

  public MultiStrategy(List<String> dicts) {
    this.dicts = dicts;
    this.dictionary = new DynamicCustomDictionary(Config.CustomDictionaryPath);
    for (String word : dicts) {
      this.dictionary.insert(word, "dt 1024");
    }
  }

  public void showAll(String text) {
    indexAnalyze(text);
    indexNormal(text);
    indexNShort(text);
  }


  private void indexNShort(String text) {
    final Segment segment = HanLP.newSegment();
    segment.enableCustomDictionary(false).enablePlaceRecognize(true).enableOrganizationRecognize(true);
    final List<Term> termList = segment.seg(text);
    System.out.println("----nshort 分词结果----");
    for (Term term : termList) {
      System.out.println(term + " [" + term.offset + ":" + (term.offset + term.word.length()) + "]");
    }

  }


  private void indexNormal(String text) {
    final Segment segment = HanLP.newSegment();
    final List<Term> termList = segment.seg(text);
    System.out.println("----标准分词结果----");
    for (Term term : termList) {
      System.out.println(term + " [" + term.offset + ":" + (term.offset + term.word.length()) + "]");
    }

  }

  /**
   * 搜索分词算法
   * @param sentence
   */
  private void indexAnalyze(String sentence) {
    final Segment segment = HanLP.newSegment();
    segment.enableIndexMode(true);
    final List<Term> termList = segment.seg(sentence);
    System.out.println("---搜索引擎分词结果----");
    for (Term term : termList) {
      System.out.println(term + " [" + term.offset + ":" + (term.offset + term.word.length()) + "]");
    }
  }


  public static void main(String[] args) {

    new MultiStrategy(Lists.newArrayList("欧美头像")).showAll(
        "欧美头"
    );
  }


  public static void showCustomOnly(String text, DynamicCustomDictionary dictionary) {
    System.err.println("--------------------");
    final Segment segment = HanLP.newSegment();
    segment.customDictionary = dictionary;
    final List<Term> seg = segment.seg(text);
    seg.forEach(System.err::println);
  }
}
