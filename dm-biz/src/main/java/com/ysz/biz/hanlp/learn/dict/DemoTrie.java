package com.ysz.biz.hanlp.learn.dict;

import java.util.HashMap;
import java.util.Map;

public class DemoTrie<V> {

  private Map<Character, DemoTrie<V>> children = new HashMap<>();


  private V value;


  public DemoTrie(V value) {
    this.value = value;
  }

  /**
   * <pre>
   *   增加一个 子节点, 生成一个父子关系
   * </pre>
   * @param childKey child Key
   * @param value child value
   * @param overWrite if true , force to overwrite childKey -> childValue
   * @return
   */
  public DemoTrie<V> addChild(
      char childKey,
      V value,
      boolean overWrite
  ) {

    DemoTrie<V> child = this.children.get(childKey);
    if (null == child) {
      child = new DemoTrie<V>(value);
      this.children.put(childKey, child);
    }

    if (overWrite) {
      this.children.put(childKey, child);
    }

    return child;
  }


  public V findValue(String word) {
    DemoTrie<V> state = this;

    for (char c : word.toCharArray()) {
      state = state.children.get(c);
      if (state == null) {
        return null;
      }
    }

    return state.value;

  }

  public void addWord(String word, V value) {
    DemoTrie<V> state = this;

    int length = word.length();
    for (int j = 0; j < length; j++) {
      char c = word.charAt(j);
      if (j != length - 1) {
        /*overwrite = false, 中间状态不能去覆盖别人已经有的词*/
        /*value = null 中间状态不能记录 value*/
        state = state.addChild(c, null, false);
      } else {
        state = state.addChild(c, value, true);
      }
    }

  }


  public static void main(String[] args) throws Exception {
    DemoTrie<String> trie = new DemoTrie<>(null);
    trie.addWord("自然", "ziran");
    trie.addWord("自然人", "ziranren");
    trie.addWord("你好", "nihao");

    System.out.println(trie.findValue("自然"));
    System.out.println(trie.findValue("自然人"));
    System.out.println(trie.findValue("你好"));
  }

}
