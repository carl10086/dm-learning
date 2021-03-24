package com.ysz.biz.ansj.trie.darts;

import com.google.common.collect.Lists;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class DartsDoubleArrayTrie_Dm_001_Basic {


  /**
   * 测试数据在这里是有序的 ...
   */
  public List<String> initWords() {
    return Lists.newArrayList(
        "一举",
        "一举一动",
        "一举成名",
        "一举成名天下知",
        "万能",
        "万能胶"
    );
  }

  public void execute() {
    final List<String> words = initWords();
    Set<Character> characterSet = new HashSet<>();
    for (String word : words) {
      for (char c : word.toCharArray()) {
        characterSet.add(c);
      }
    }

    System.out.println("字典词条：" + words.size());
    final DartsDoubleArrayTrie dat = new DartsDoubleArrayTrie();
    System.out.println("是否错误: " + dat.build(words));
    List<Integer> integerList = dat.commonPrefixSearch("一举成名天下知");
    for (int index : integerList) {
      System.out.println(words.get(index));
    }

  }

  public static void main(String[] args) {
    DartsDoubleArrayTrie_Dm_001_Basic bootStrap = new DartsDoubleArrayTrie_Dm_001_Basic();
    bootStrap.execute();
  }

}
