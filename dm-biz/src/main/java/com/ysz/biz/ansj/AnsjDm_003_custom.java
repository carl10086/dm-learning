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
    DicLibrary.insert(DicLibrary.DEFAULT, "攻城狮", "我是词性", 1000);
    DicLibrary.insert(DicLibrary.DEFAULT, "单身狗", "我是词性", 1000);
    DicLibrary.insert(DicLibrary.DEFAULT, "狮逆", "我是词性", 1000);
    DicLibrary.insert(DicLibrary.DEFAULT, "娶白", "我是词性", 1000);
    DicLibrary.insert(DicLibrary.DEFAULT, "白富美", "我是词性", 1000);
    Result parse = DicAnalysis.parse("攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰，习近平");
    for (Term term : parse) {
      System.out.println(term);
    }
  }

}
