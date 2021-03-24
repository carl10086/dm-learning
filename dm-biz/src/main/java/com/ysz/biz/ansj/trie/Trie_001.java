package com.ysz.biz.ansj.trie;

/**
 * 最简单的 trie 实现 ...
 */
public class Trie_001 {

  private TrieNode root = new TrieNode('/');

  public void insertString(String src) {
    if (src != null) {
      insert(src.toLowerCase().toCharArray());
    }
  }

  // 往 Trie 树插入一个字符串 .....
  private void insert(char[] text) {
    TrieNode p = root;

    for (int i = 0; i < text.length; ++i) {
      // 当前的字符
      final char theChar = text[i];
      // 当前字符对应的位置. 26个字符不考虑大小写 -a 就能得到
      final int index = theChar - 'a';
      // 对应的 p 对应的 child
      final TrieNode child = p.children[index];
      if (child == null) { // child 为空, 就增加一个
        TrieNode newNode = new TrieNode(theChar); // 都为空了、就要创建一个新的
        p.children[index] = newNode; // 新的放到对应的位置上、下次来判断就是 != null
      }
      p = p.children[index]; // 下一个字符、从这里开始 ..
    }
    p.isEndingChar = true;
  }

  // 在 Trie 树中查找一个字符串 .....
  public boolean find(char[] pattern) {
    TrieNode p = root;
    for (int i = 0; i < pattern.length; ++i) {
      int index = pattern[i] - 'a';
      if (p.children[index] == null) {
        return false;
      }
      p = p.children[index];
    }

    if (p.isEndingChar == false) {
      return false; // 不能完全匹配、只是前缀
    } else {
      return true; // 找到 pattern
    }
  }

  public class TrieNode {

    public char data;
    public TrieNode[] children = new TrieNode[26]; // 这里考虑的是最多 26个字符 ....往往这里不考虑大小写 ....
    public boolean isEndingChar = false;

    public TrieNode(final char data) {
      this.data = data;
    }
  }


  public static void main(String[] args) {
    // 测试逻辑
    Trie_001 trie001 = new Trie_001();
    insertAndFind("how", trie001);
    tryFind("how", trie001);
    tryFind("ho", trie001);
  }

  public static void tryFind(String src, Trie_001 trie) {
    System.out.printf("find %s -> result %s\n", src, trie.find(src.toLowerCase().toCharArray()));
  }

  public static void insertAndFind(String src, Trie_001 trie) {
    final char[] chars = src.toLowerCase().toCharArray();
    trie.insert(chars);
    if (!trie.find(chars)) {
      System.err.println("error, can't find :" + src);
    }

  }
}
