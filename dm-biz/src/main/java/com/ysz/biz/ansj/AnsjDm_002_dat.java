package com.ysz.biz.ansj;

import java.io.BufferedReader;
import java.io.StringReader;
import org.nlpcn.commons.lang.tire.GetWord;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.library.Library;

public class AnsjDm_002_dat {

  public static void main(String[] args) throws Exception {
//    DoubleArrayTire dat = DoubleArrayTire
//        .loadText(DicReader.getInputStream("core.dic"), AnsjItem.class);

    String dic =
        "中国\t1\tzg\n" +
            "人名\t2\n" +
            "中国人民\t4\n" +
            "人民\t3\n" +
            "孙健\t5\n" +
            "CSDN\t6\n" +
            "java\t7\n" +
            "java学习\t10\n";
    Forest forest = Library.makeForest(new BufferedReader(new StringReader(dic)));

    /**
     * 删除一个单词
     */
    Library.removeWord(forest, "中国");
    /**
     * 增加一个新词
     */
    Library.insertWord(forest, "中国人");
    String content = "中国人名识别是中国人民的一个骄傲.孙健人民在CSDN中学到了很多最早iteye是java学习笔记叫javaeye但是java123只是一部分";
    GetWord udg = forest.getWord(content);

    String temp = null;
    while ((temp = udg.getFrontWords()) != null) {
      System.out.println(temp + "\t\t" + udg.getParam(1) + "\t\t" + udg.getParam(2));
    }
  }
}
