package com.ysz.biz.hanlp.split;

import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;
import java.util.List;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/28
 **/
public class StandardTokenizerDm {


  public static void main(String[] args) {
    List<Term> termList = StandardTokenizer.segment("商品和服务");
    System.out.println(termList);
  }

}
