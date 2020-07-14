package com.ysz.dm.es.ansj;

import org.ansj.library.StopLibrary;
import org.ansj.recognition.impl.StopRecognition;
import org.ansj.splitWord.analysis.DicAnalysis;

/**
 * @author carl
 */
public class Ansj_Stop_Dm_002 {


  private static final String KEY = StopLibrary.DEFAULT;


  private Ansj_Stop_Dm_002 init() {
    if (StopLibrary.get(KEY) == null) {
      StopLibrary.put(KEY, KEY, new StopRecognition());
    }
    return this;
  }


  public void test() {
    System.out.println(DicAnalysis.parse("我们在做什么"));
    StopLibrary.insertStopWords(KEY, "在");
    StopRecognition stopRecognition = StopLibrary.get(KEY);
    DicAnalysis dicAnalysis = new DicAnalysis();
    System.out.println(DicAnalysis.parse("我们在做什么").recognition(stopRecognition));
  }

  public static void main(String[] args) {
    new Ansj_Stop_Dm_002().init().test();
  }

}
