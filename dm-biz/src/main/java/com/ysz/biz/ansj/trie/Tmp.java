package com.ysz.biz.ansj.trie;

import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.Collections;

public class Tmp {

  public static void main(String[] args) {
    ArrayList<String> strings = Lists.newArrayList("a", "b", "c", "aa", "ab", "ac");
    Collections.sort(strings);
    System.err.println(strings);

  }

}
