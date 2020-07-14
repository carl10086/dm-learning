package com.ysz.dm.es.ansj;

import java.util.List;
import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.BaseAnalysis;
import org.ansj.splitWord.analysis.DicAnalysis;
import org.ansj.splitWord.analysis.IndexAnalysis;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.ansj.splitWord.analysis.ToAnalysis;

/**
 * @author carl
 */
public class Ansj_Basic_Dm_001 {


  public static void main(String[] args) throws Exception {
    String str = "洁面仪配合洁面深层清洁毛孔 清洁鼻孔面膜碎觉使劲挤才能出一点点皱纹 脸颊毛孔修复的看不见啦 草莓鼻历史遗留问题没辙 脸和脖子差不多颜色的皮肤才是健康的 长期使用安全健康的比同龄人显小五到十岁 28岁的妹子看看你们的鱼尾纹";

    System.out.println(BaseAnalysis.parse(str));
    System.out.println(ToAnalysis.parse(str));
    System.out.println(DicAnalysis.parse(str));
    System.out.println(IndexAnalysis.parse(str));
    System.out.println(NlpAnalysis.parse(str));
    Result parse = ToAnalysis.parse("发生了什么/东西");
    List<Term> terms = parse.getTerms();
    for (Term term : terms) {
      System.out.println(term.getNatureStr());
   }
  }

}
