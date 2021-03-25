package com.ysz.biz.ansj;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.library.DicLibrary;
import org.ansj.splitWord.analysis.DicAnalysis;
import org.nlpcn.commons.lang.tire.domain.Forest;

public class AnsjDm_003_custom {

  public static void main(String[] args) {
    Forest forest = DicLibrary.get();
    if (forest == null) {
      // 创建一个 根节点 ...
      DicLibrary.put(DicLibrary.DEFAULT, DicLibrary.DEFAULT, new Forest());
    }
    DicLibrary.insert(DicLibrary.DEFAULT, "增加新词", "我是词性", 1000);
    DicLibrary.insert(DicLibrary.DEFAULT, "增加新词", "我是词性2", 1000);
    Result parse = DicAnalysis.parse("这是用户自定义词典增加新词的例子");
    System.out.println(parse);
    boolean flag = false;
    for (Term term : parse) {
      flag = flag || "增加新词".equals(term.getName());
    }
  }

}
