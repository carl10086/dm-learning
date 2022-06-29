package com.ysz.biz.hanlp.learn.dict.bin;

public class BinTrie<V> extends BaseNode<V> {

  private int size;

  @Override
  public BaseNode<V> child(char c) {
    return null;
  }

  public BinTrie() {
    children = new BaseNode[65535 + 1];
    size = 0;
    status = Status.NOT_WORD_1;
  }


  public void put(String key, V value) {
    if (key.length() == 0) {
      return;
    }

    BaseNode<V> branch = this;
    char[] chars = key.toCharArray();

    for (int i = 0; i < chars.length - 1; i++) {
      char aChar = chars[i];


    }

  }

  @Override
  public boolean addChild(BaseNode<V> node) {
    return false;
  }
}
