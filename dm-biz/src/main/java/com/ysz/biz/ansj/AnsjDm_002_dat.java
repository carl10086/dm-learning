package com.ysz.biz.ansj;

import org.ansj.dic.DicReader;
import org.ansj.domain.AnsjItem;
import org.nlpcn.commons.lang.dat.DoubleArrayTire;

public class AnsjDm_002_dat {

  public static void main(String[] args) throws Exception {
    DoubleArrayTire dat = DoubleArrayTire
        .loadText(DicReader.getInputStream("core.dic"), AnsjItem.class);
  }
}
